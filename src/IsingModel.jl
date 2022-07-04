using LinearAlgebra, IfElse

module IsingModel

mutable struct IsingModelInt
    Cells::Tuple{Vararg{Int,N} where N}
    Steps::Int
    #SaveStep::Int
    State::Array{Int8}
    β::Float64
end

mutable struct IsingModelBit
    Cells::Tuple{Vararg{Int,N} where N}
    Steps::Int
    #SaveStep::Int
    State::BitArray
    β::Float64
end

function SerialStepInt!(m::IsingModelInt, Cells::Tuple{Int}, temp::Array{Int8,1})
    h_min = -2
    h_max = 2
    prob = [1/(1+exp(-2*m.β*h)) for h ∈ h_min:h_max]
    @tturbo for i ∈ eachindex(m.State)
        #left = s[i == 1 ? n : i-1]
        #right = s[i == n ? 1 : i+1]
        h = m.State[i == 1 ? n : i-1] + m.State[i == n ? 1 : i+1]
        si_old = m.State[i]
        m.State[i] = rand(Float64) < prob[h-h_min+1] ? 1 : -1
    end
end

function EvaluateModel(m::IsingModel)
    return nothing
end

end # module
