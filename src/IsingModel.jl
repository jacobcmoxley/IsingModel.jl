module IsingModel

using Distributed, DistributedArrays, LinearAlgebra, DelimitedFiles, Random, LoopVectorization

export Ising, SerialStep!, EvaluateModel!

"""
## Constructors
Ising - Type
Ising(Cells::Tuple{Vararg{Int,N} where N}, 
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64)

Instante an Ising object. Steps is the total number of steps. SaveStep is how often the state is saved. SaveFile is the absolute file path. β is temperature

### Examples
```
Ising((15,),(2,),100,10,"~/Documents/IsingExample1.txt",20.2)
```
Object which will have 30 cells, 2 workers, take 100 total steps, save on step 10, 20, ..., 100 at file ~/Documents/IsingExample1.txt
"""
mutable struct Ising
    Cells::Tuple{Vararg{Int,N} where N}
    Procs::Tuple{Vararg{Int,N} where N}
    Steps::Int
    SaveStep::Int
    SaveFile::String
    β::Float64
    State::Union{Array{Int8},DArray{Int8}}
    function Ising(Cells::Tuple{Vararg{Int,N} where N},
        Procs::Tuple{Vararg{Int,N} where N},
        Steps::Int,
        SaveStep::Int,
        SaveFile::String,
        β::Float64,
        State::Union{Array{Int8},DArray{Int8}})
        if nworkers() < prod(b)
            addprocs(nworkers()-prod(b))
        end
        #@everywhere using DistributedArrays

        if prod(b) == 1
            Ising(Cells,Procs,Steps,SaveStep,SaveFile,β) = new(Cells,Procs,Steps,SaveStep,SaveFile,β,rand(Int8[-1,1],Cells))
        else
            @everywhere f(x)::Int8 = x<0.5 ? -1 : 1
            Ising(Cells,Procs,Steps,SaveStep,SaveFile,β) = new(Cells,Procs,Steps,SaveStep,SaveFile,β,
                map(f,drand(Cells .* Procs,workers()[1:prod(Procs)], Procs)))
        end
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

function SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,2})
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

function SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,3})
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

function InitDist(m::Ising)
    return 1
end

function EvaluateModel!(m::Ising, StepFunction::Function)
    f = open(m.SaveFile,"w")
    writedlm(f, m.State)
    temp = similar(m.State)
    for st ∈ 1:m.Steps 
        m.State = StepFunction(m, m.Cells, temp)
        if st % m.SaveStep == 0
            writedlm(f,m.State)
        end
    end
    close(f)
end

end # module
