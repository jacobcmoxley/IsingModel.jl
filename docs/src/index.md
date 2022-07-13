```@docs
Ising(Cells::Tuple{Vararg{Int,N} where N}, 
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64)
SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,1})
```