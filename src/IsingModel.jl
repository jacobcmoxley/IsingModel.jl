module IsingModel

using LinearAlgebra, DelimitedFiles, Random, LoopVectorization

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
        #si_old = m.State[i]
        temp[i] = rand(Float64) < prob[h-h_min+1] ? 1 : -1
    end
    return temp
end

#=
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
=#

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
