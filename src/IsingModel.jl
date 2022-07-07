module IsingModel

using LinearAlgebra, DelimitedFiles, Random, LoopVectorization, Distributed

export Ising, SerialStep!, EvaluateModel!

mutable struct Ising
    Cells::Tuple{Vararg{Int,N} where N}
    Steps::Int
    SaveStep::Int
    SaveFile::String
    State::Array{Int8}
    β::Float64
    Ising(Cells,Steps,SaveStep,SaveFile,β) = new(Cells,Steps,SaveStep,SaveFile,rand(Int8[-1,1],Cells),β)
end

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

function InitDist()
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
