```@meta
DocTestSetup = :(using DistributedArrays)
```

```@docs
Ising(Cells::Tuple{Vararg{Int,N} where N},
    Procs::Tuple{Vararg{Int,N} where N},
    Steps::Int,
    SaveStep::Int,
    SaveFile::String,
    β::Float64,
    State::Union{Array{Int8},DArray{Int8}})
```

```@docs
SerialStep!(m::Ising, Cells::Tuple{Int}, temp::Array{Int8,1})
```