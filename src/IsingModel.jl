using LinearAlgebra, IfElse, Random

module IsingModel

mutable struct IsingModelInt
    Cells::Tuple{Vararg{Int,N} where N}
    Steps::Int
    SaveStep::Int
    State::Array{Int8}
    β::Float64
end

function SerialStepInt!(m::IsingModelInt, Cells::Tuple{Int}, temp::Array{Int8,1})
    h_min = -2
    h_max = 2
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:h_max]
    n = length(m.State)
    @tturbo for i ∈ eachindex(m.State)
        #left = s[i == 1 ? n : i-1]
        #right = s[i == n ? 1 : i+1]
        h = m.State[i == 1 ? n : i-1] + m.State[i == n ? 1 : i+1]
        #si_old = m.State[i]
        temp[i] = rand(Float64) < prob[h-h_min+1] ? 1 : -1
    end
    return temp
end

"""
mutable struct IsingModelBit
    Cells::Tuple{Vararg{Int,N} where N}
    Steps::Int
    SaveStep::Int
    State::BitArray
    β::Float64
end

function 1dBitLog(A)
    n = length(A)
    for i ∈ eachindex(A)
        left = A[i==1 ? n : i-1]
        right = A[i==n ? 1 : i+1]
        if (left && right)
            h = 1
        elseif (left ⊻ right)
            h = 2
        else
            h = 3
        end
    end
end

function SerialStepBit!(m::IsingModelBit, Cells::Tuple{Int}, temp::BitArray{1})
    h_min = -2
    h_max = 2
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:2:h_max]
    n = length(m.State)
    @inbounds @simd for i ∈ eachindex(m.State)
        left = m.State[i==1 ? n : i-1]
        right = m.State[i==n ? 1 : i+1]
        if left & right
            h = 1
        elseif left ⊻ right
            h = 2
        else
            h = 3
        end
        temp[i] = rand(Float64) < prob[h] ? 1 : 0
    end
    return temp
end
"""

function EvaluateModel(m::IsingModel, step::function)
    
end

end # module
