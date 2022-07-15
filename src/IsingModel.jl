module IsingModel

using Distributed, DistributedArrays, LinearAlgebra, DelimitedFiles, Random, LoopVectorization
@eval @everywhere using Distributed, DistributedArrays, LinearAlgebra, DelimitedFiles, Random, LoopVectorization

export Ising, SerialStep!, DistStep, DistRule, EvaluateModel!

mutable struct Ising
    Cells::Tuple{Vararg{Int,N} where N}
    Procs::Tuple{Vararg{Int,N} where N}
    Steps::Int
    SaveStep::Int
    SaveFile::String
    β::Float64
    State::Union{Array{Int8},DArray{Int8}}
end

"""
```
Ising(Cells::Tuple{Vararg{Int,N} where N}, 
    Procs::Truple{Vararg{Int,N} where N}
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64,
    State::Union{Array{Int8},DArray{Int8}})
```

#Arguments
Cells - Tuple of the number of cells per processor. Note all processors must have the same number of cells.
Procs - Tuple of the number of processors along the same axis as the cells.
Steps - The total number of steps the simulation will take
SaveStep - The number of steps in between saving the State
SaveFile - Absolute file path where the state will be saved
β - Temperature
State - Optional argument for initial state. If not provided it will be initialized to a uniform random.

# Examples
```
Ising((15,),(2,),100,10,"~/Documents/IsingExample1.txt",20.2)
```
Object which will have 30 cells, 2 workers, take 100 total steps, save on step 10, 20, ..., 100 at file ~/Documents/IsingExample1.txt
"""
function Ising(Cells::Tuple{Vararg{Int,N} where N},
    Procs::Tuple{Vararg{Int,N} where N},
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64)
    if nworkers() < prod(Procs)
        addprocs(prod(Procs) - nworkers() +1)
        @eval @everywhere using Distributed, DistributedArrays, LinearAlgebra, DelimitedFiles, Random, LoopVectorization
        @eval export DistStep
    end

    if prod(Procs) == 1
        return Ising(Cells,Procs,StepFunction,Steps,SaveStep,SaveFile,β,rand(Int8[-1,1],Cells))
    else
        @eval @everywhere UpDown(x)::Int8 = x<0.5 ? -1 : 1
        return Ising(Cells,Procs,StepFunction,Steps,SaveStep,SaveFile,β,
            map(UpDown,drand(Cells .* Procs,workers()[1:prod(Procs)], Procs)))
    end
end

function Ising(Cells::Tuple{Vararg{Int,N} where N},
    Procs::Tuple{Vararg{Int,N} where N},
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64,
    State::Union{Array{Int8},DArray{Int8}})
    if nworkers() < prod(Procs)
        addprocs(prod(Procs) - nworkers()+1)
        @eval @everywhere using Distributed, DistributedArrays, LinearAlgebra, DelimitedFiles, Random, LoopVectorization
        @eval export DistStep
    end

    if prod(Procs) == 1
        return Ising(Cells,Procs,StepFunction,Steps,SaveStep,SaveFile,β,State)
    else
        @eval @everywhere UpDown(x)::Int8 = x<0.5 ? -1 : 1
        return Ising(Cells,Procs,StepFunction,Steps,SaveStep,SaveFile,β,State)
    end
end

"""
SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,1})

Take one step forward on a single processor. The innermost loop will be threaded and simd using the tturbo macro.

### Arguments
m::Ising an Ising object

Cells::Tuple{Vararg{Int,N}} the Cells field of the Ising object m. Included to specialize the function for different dimensions

temp::Array{Int8,N} preallocated array, similar(m.State)
"""
function SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,1})
    h_min = -2
    h_max = 2
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:h_max]
    n = length(m.State)
    @tturbo for i ∈ eachindex(m.State)
        #left = s[i == 1 ? n : i-1]
        #right = s[i == n ? 1 : i+1]
        h = m.State[i == 1 ? n : i-1] + m.State[i == n ? 1 : i+1]
        temp[i] = rand(Float64) < prob[h-h_min+1] ? 1 : -1
    end
    return temp
end

function SerialStep!(m::Ising, Cells::Tuple{Int,Int}, temp::Array{Int8,2})
    h_min = -4
    h_max = 4
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:h_max]
    m, n = size(m.State)
    @inbounds for j ∈ 1:n
        @tturbo for i ∈ 1:m
            top = m.State[i == 1 ? m : i-1, j]
            bottom = m.State[i == m ? 1 : i+1, j]
            right = m.State[i, j == 1 ? n : j-1]
            left = m.State[i, j == n ? 1 : j+1]
            h = top + bottom + right + left
            temp[i,j] = rand(Float64) < prob[h-h_min+1] ? +1 : -1
        end
    end
    return temp
end

function SerialStep!(m::Ising, Cells::Tuple{Int,Int,Int}, temp::Array{Int8,3})
    h_min = -6
    h_max = 6
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:h_max]
    m, n, o = length(m.State)
    @inbounds for k ∈ 1:o
        @inbounds for j ∈ 1:n
            @tturbo for i ∈ 1:m
                front = m.State[i, j, k == 1 ? o : k-1]
                back = m.State[i, j, k == o ? 1 : k+1]
                top = m.State[i == 1 ? m : i-1, j, k]
                bottom = m.State[i == m ? 1 : i+1, j, k]
                right = m.State[i, j == 1 ? n : j-1, k]
                left = m.State[i, j == n ? 1 : j+1, k]
                h = top + bottom + right + left + front + back
                temp[i,j,k] = rand(Float64) < prob[h-h_min+1] ? +1 : -1
            end
        end
    end
    return temp
end



function DistStep(m::Ising, Cells::Tuple{Int,Int}, Procs::Tuple{Int,Int})
    DArray(size(m.State),procs(m.State)) do I
        top   = mod(first(I[1])-2,size(d,1))+1
        bot   = mod( last(I[1])  ,size(d,1))+1
        left  = mod(first(I[2])-2,size(d,2))+1
        right = mod( last(I[2])  ,size(d,2))+1


        old = Array{Int8}(undef, length(I[1])+2, length(I[2])+2)
        old[1      , 1      ] = d[top , left]   # left side
        old[2:end-1, 1      ] = d[I[1], left]
        old[end    , 1      ] = d[bot , left]
        old[1      , 2:end-1] = d[top , I[2]]
        old[2:end-1, 2:end-1] = d[I[1], I[2]]   # middle
        old[end    , 2:end-1] = d[bot , I[2]]
        old[1      , end    ] = d[top , right]  # right side
        old[2:end-1, end    ] = d[I[1], right]
        old[end    , end    ] = d[bot , right]

        DistRule(old)
    end
end

function DistRule(old::Array{Int8,2})
    m, n = size(old)
    new = similar(old, m-2, n-2)
    h_min = -4
    h_max = 4
    prob = [1/(1+exp(-2*20.2*h)) for h ∈ h_min:h_max]
    @inbounds for j ∈ 2:n-1
        @tturbo for i ∈ 2:m-1
            h = +(old[i-1,j], old[i+1,j], old[i,j-1], old[i,j+1])
            new[i-1,j-1] = rand(Float64) < prob[h-h_min+1] ? +1 : -1
        end
    end
    new
end

function EvaluateModel!(m::Ising, StepFunction::Function)
    f = open(m.SaveFile,"w")
    writedlm(f, m.State)
    temp = similar(m.State) #this might need to change
    for st ∈ 1:m.Steps 
        m.State = StepFunction(m, m.Cells, temp)
        if st % m.SaveStep == 0
            writedlm(f,m.State)
        end
    end
    close(f)
end

end # module
