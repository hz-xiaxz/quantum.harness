module RGRecursiveSparseSOHS

using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools

const MOI = JuMP.MOI
const N = 12
const BLOCK_SIZE = 3
const N_BLOCKS = 4
const Word12 = NTuple{N,UInt8}
const Word4 = NTuple{N_BLOCKS,UInt8}
const PAULIS = (
    ComplexF64[1 0; 0 1],
    ComplexF64[0 1; 1 0],
    ComplexF64[0 -im; im 0],
    ComplexF64[1 0; 0 -1],
)

"Metadata fixes the finite problem and Wang et al. Eq. (4.5) basis choice."
const PROTOTYPE_METADATA = Dict{String,Any}(
    "prototype" => "minimal RG-recursive sparse SOHS certificate",
    "N" => N,
    "boundary" => "periodic",
    "spin" => "1/2",
    "hamiltonian" => "H = (1/4) sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})",
    "ed_sector" => "total Sz = 0",
    "physical_basis" => "Wang-style 1D contiguous-support B_1 from Eq. (4.5): identity plus every one-site X/Y/Z Pauli string",
    "physical_basis_order" => 1,
    "physical_basis_dimension" => 1 + 3N,
    "coarse_partition" => "four consecutive open three-site blocks: (1:3),(4:6),(7:9),(10:12)",
    "coarse_basis" => "the same B_1 construction on four logical spins",
    "coarse_basis_dimension" => 1 + 3N_BLOCKS,
    "rg_depth" => 1,
    "map_status" => "fixed before optimization; resulting problem is an SDP",
    "operator_representation" => "sparse Dict{NTuple{12,UInt8}, coefficient}; no global 4096x4096 matrix",
)

zero_word(::Val{L}) where {L} = ntuple(_ -> UInt8(0), L)

function single_word(::Val{L}, site::Int, axis::Int) where {L}
    1 <= site <= L || throw(BoundsError(1:L, site))
    1 <= axis <= 3 || throw(BoundsError(1:3, axis))
    ntuple(i -> UInt8(i == site ? axis : 0), L)
end

"Wang-style periodic contiguous-support basis through order `max_support`."
function wang_basis(::Val{L}; max_support::Int=1) where {L}
    1 <= max_support <= L || throw(ArgumentError("max_support must lie in 1:$L"))
    basis = [zero_word(Val(L))]
    for support in 1:max_support, start in 1:L
        sites = ntuple(offset -> mod1(start + offset - 1, L), support)
        for axes in Iterators.product(ntuple(_ -> 1:3, support)...)
            push!(basis, ntuple(site -> begin
                offset = findfirst(==(site), sites)
                isnothing(offset) ? UInt8(0) : UInt8(axes[offset])
            end, L))
        end
    end
    unique(basis)
end

const LOCAL_PRODUCT = let
    table = Matrix{Tuple{ComplexF64,UInt8}}(undef, 4, 4)
    for a in 0:3, b in 0:3
        M = PAULIS[a + 1] * PAULIS[b + 1]
        for c in 0:3
            coefficient = tr(PAULIS[c + 1]' * M) / 2
            if abs(coefficient) > 1e-12
                table[a + 1, b + 1] = (coefficient, UInt8(c))
                break
            end
        end
    end
    table
end

function multiply_words(left::NTuple{L,UInt8}, right::NTuple{L,UInt8}) where {L}
    phase = 1.0 + 0.0im
    labels = Vector{UInt8}(undef, L)
    for i in 1:L
        local_phase, labels[i] = LOCAL_PRODUCT[Int(left[i]) + 1, Int(right[i]) + 1]
        phase *= local_phase
    end
    phase, Tuple(labels)
end

function add_coefficient!(dictionary::Dict{K,V}, key::K, value) where {K,V}
    dictionary[key] = get(dictionary, key, zero(V)) + value
    dictionary
end

function heisenberg_dictionary()
    h = Dict{Word12,ComplexF64}()
    for site in 1:N
        neighbor = mod1(site + 1, N)
        for axis in 1:3
            word = ntuple(i -> UInt8(i == site || i == neighbor ? axis : 0), N)
            add_coefficient!(h, word, 0.25 + 0im)
        end
    end
    h
end

function canonical_phase!(vector::AbstractVector)
    pivot = findfirst(x -> abs(x) > 1e-12, vector)
    isnothing(pivot) && error("zero eigenvector")
    vector .*= exp(-im * angle(vector[pivot]))
    real(vector[pivot]) < 0 && (vector .*= -1)
    vector
end

"Three-site open-chain ground doublet isometry J: logical spin -> physical block."
function block_isometry()
    I2, X, Y, Z = PAULIS
    h3 = (kron(X, X, I2) + kron(Y, Y, I2) + kron(Z, Z, I2) +
          kron(I2, X, X) + kron(I2, Y, Y) + kron(I2, Z, Z)) / 4
    sz3 = (kron(Z, I2, I2) + kron(I2, Z, I2) + kron(I2, I2, Z)) / 2
    J = zeros(ComplexF64, 8, 2)
    for (column, magnetization) in enumerate((0.5, -0.5))
        sector = findall(i -> isapprox(real(sz3[i, i]), magnetization; atol=1e-12), 1:8)
        decomposition = eigen(Hermitian(h3[sector, sector]))
        vector = ComplexF64.(decomposition.vectors[:, argmin(decomposition.values)])
        canonical_phase!(vector)
        J[sector, column] = vector
    end
    J
end

struct FixedBlockChannel
    J::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    kraus::Vector{Matrix{ComplexF64}}
    pauli_images::NTuple{4,Dict{NTuple{3,UInt8},ComplexF64}}
end

function pauli_decomposition3(matrix::AbstractMatrix; tolerance=1e-12)
    size(matrix) == (8, 8) || throw(DimensionMismatch("three-site operator must be 8x8"))
    result = Dict{NTuple{3,UInt8},ComplexF64}()
    for labels in Iterators.product(ntuple(_ -> UInt8(0):UInt8(3), 3)...)
        word = Tuple(labels)
        P = kron((PAULIS[Int(label) + 1] for label in word)...)
        coefficient = tr(P' * matrix) / 8
        abs(coefficient) > tolerance && (result[word] = coefficient)
    end
    result
end

"Build Φ(X)=JXJ†+Tr((I/2)X)Q and Kraus operators satisfying Σ K K†=I_8."
function fixed_block_channel()
    J = block_isometry()
    Q = Matrix{ComplexF64}(I, 8, 8) - J * J'
    decomposition = eigen(Hermitian(Q))
    complement = [ComplexF64.(decomposition.vectors[:, i]) for i in eachindex(decomposition.values)
                  if decomposition.values[i] > 0.5]
    kraus = Matrix{ComplexF64}[J]
    for q in complement, logical in 1:2
        K = zeros(ComplexF64, 8, 2)
        K[:, logical] = q / sqrt(2)
        push!(kraus, K)
    end
    image(X) = J * X * J' + (tr(X) / 2) * Q
    images = ntuple(i -> pauli_decomposition3(image(PAULIS[i])), 4)
    FixedBlockChannel(J, Q, kraus, images)
end

apply_channel(channel::FixedBlockChannel, X::AbstractMatrix) =
    channel.J * X * channel.J' + (tr(X) / 2) * channel.Q

function lift_coarse_word(channel::FixedBlockChannel, word::Word4)
    partial = Dict{Tuple{Vararg{UInt8}},ComplexF64}(() => 1.0 + 0im)
    for block in 1:N_BLOCKS
        following = Dict{Tuple{Vararg{UInt8}},ComplexF64}()
        for (prefix, a) in partial, (local_word, b) in channel.pauli_images[Int(word[block]) + 1]
            key = (prefix..., local_word...)
            following[key] = get(following, key, 0.0 + 0im) + a * b
        end
        partial = following
    end
    Dict{Word12,ComplexF64}(Tuple(key) => value for (key, value) in partial if abs(value) > 1e-12)
end

function gram_dictionary(G, basis)
    K = eltype(basis)
    coefficients = Dict{K,Any}()
    for i in eachindex(basis), j in eachindex(basis)
        phase, word = multiply_words(basis[i], basis[j])
        coefficients[word] = get(coefficients, word, 0.0 + 0im) + phase * G[i, j]
    end
    coefficients
end

function lifted_gram_dictionary(R, basis, channel::FixedBlockChannel)
    coefficients = Dict{Word12,Any}()
    cache = Dict{Word4,Dict{Word12,ComplexF64}}()
    for i in eachindex(basis), j in eachindex(basis)
        phase, coarse_word = multiply_words(basis[i], basis[j])
        image = get!(cache, coarse_word) do
            lift_coarse_word(channel, coarse_word)
        end
        for (physical_word, image_coefficient) in image
            coefficients[physical_word] = get(coefficients, physical_word, 0.0 + 0im) +
                phase * image_coefficient * R[i, j]
        end
    end
    coefficients
end

struct CertificateProblem
    model::JuMP.Model
    lambda
    physical_gram
    coarse_gram
    coefficient_expressions::Dict{Word12,Any}
    hamiltonian::Dict{Word12,ComplexF64}
    metadata::Dict{String,Any}
end

function add_scalar_equality!(model, expression, constraints)
    if expression isa Number
        abs(expression) <= 1e-10 || error("inconsistent constant coefficient: $expression")
    else
        push!(constraints, @constraint(model, expression == 0))
    end
end

"Build baseline or one-level RG certificate; the fixed map leaves only affine equalities and PSD cones."
function build_certificate(; rg::Bool, physical_order::Int=1, coarse_order::Int=1,
        optimizer=MosekTools.Optimizer, silent::Bool=true)
    model = Model(optimizer)
    silent && set_silent(model)
    physical_basis = wang_basis(Val(N); max_support=physical_order)
    @variable(model, G[1:length(physical_basis), 1:length(physical_basis)] in HermitianPSDCone())
    @variable(model, lambda)
    expressions = gram_dictionary(G, physical_basis)
    channel = fixed_block_channel()
    coarse_basis = wang_basis(Val(N_BLOCKS); max_support=coarse_order)
    R = nothing
    if rg
        @variable(model, Rvar[1:length(coarse_basis), 1:length(coarse_basis)] in HermitianPSDCone())
        R = Rvar
        lifted = lifted_gram_dictionary(R, coarse_basis, channel)
        for (word, expression) in lifted
            expressions[word] = get(expressions, word, 0.0 + 0im) + expression
        end
    end
    hamiltonian = heisenberg_dictionary()
    identity_word = zero_word(Val(N))
    keys_to_match = union(keys(expressions), keys(hamiltonian), (identity_word,))
    constraints = Any[]
    for word in keys_to_match
        difference = get(expressions, word, 0.0 + 0im) - get(hamiltonian, word, 0.0 + 0im)
        word == identity_word && (difference += lambda)
        add_scalar_equality!(model, real(difference), constraints)
        add_scalar_equality!(model, imag(difference), constraints)
    end
    @objective(model, Max, lambda)
    dimensions = rg ? [length(physical_basis), length(coarse_basis)] : [length(physical_basis)]
    metadata = copy(PROTOTYPE_METADATA)
    metadata["variant"] = rg ? "B$(physical_order) + one-level B$(coarse_order) block certificate-RG" :
        "baseline B$(physical_order) sparse SOHS"
    metadata["physical_basis_order"] = physical_order
    metadata["physical_basis_dimension"] = length(physical_basis)
    metadata["coarse_basis_order"] = coarse_order
    metadata["coarse_basis_dimension"] = length(coarse_basis)
    metadata["psd_block_dimensions"] = dimensions
    metadata["real_scalar_variables"] = 1 + sum(abs2, dimensions)
    metadata["coefficient_equalities"] = length(constraints)
    metadata["baseline_included_verbatim"] = true
    metadata["rg_zero_face_is_baseline"] = rg
    metadata["fixed_channel_kraus_count"] = length(channel.kraus)
    CertificateProblem(model, lambda, G, R, expressions, hamiltonian, metadata)
end

function certificate_residual(problem::CertificateProblem)
    identity_word = zero_word(Val(N))
    words = union(keys(problem.coefficient_expressions), keys(problem.hamiltonian), (identity_word,))
    maximum((abs(value(get(problem.coefficient_expressions, word, 0.0 + 0im)) -
                         get(problem.hamiltonian, word, 0.0 + 0im) +
                         (word == identity_word ? value(problem.lambda) : 0.0))
             for word in words); init=0.0)
end

function solve_certificate!(problem::CertificateProblem)
    optimize!(problem.model)
    status = termination_status(problem.model)
    status == MOI.OPTIMAL || error("Mosek failed with status $status")
    Dict{String,Any}(
        "variant" => problem.metadata["variant"],
        "E_lb" => objective_value(problem.model),
        "E_lb_per_site" => objective_value(problem.model) / N,
        "psd_block_dimensions" => problem.metadata["psd_block_dimensions"],
        "real_scalar_variables" => problem.metadata["real_scalar_variables"],
        "coefficient_equalities" => problem.metadata["coefficient_equalities"],
        "solve_time_seconds" => solve_time(problem.model),
        "coefficient_residual" => certificate_residual(problem),
        "termination_status" => string(status),
    )
end

"Sparse exact Sz=0 Hamiltonian (dimension binomial(12,6)=924), diagonalized in that sector."
function exact_diagonalization()
    states = [state for state in 0:(1 << N) - 1 if count_ones(state) == N ÷ 2]
    index = Dict(state => i for (i, state) in enumerate(states))
    rows, columns, values = Int[], Int[], Float64[]
    for (column, state) in enumerate(states)
        diagonal = 0.0
        for site in 0:N-1
            neighbor = (site + 1) % N
            left = (state >> site) & 1
            right = (state >> neighbor) & 1
            diagonal += left == right ? 0.25 : -0.25
            if left != right
                flipped = state ⊻ (1 << site) ⊻ (1 << neighbor)
                push!(rows, index[flipped]); push!(columns, column); push!(values, 0.5)
            end
        end
        push!(rows, column); push!(columns, column); push!(values, diagonal)
    end
    H = sparse(rows, columns, values, length(states), length(states))
    decomposition = eigen(Hermitian(Matrix(H)), 1:1)
    energy = real(only(decomposition.values))
    (; energy, energy_per_site=energy / N,
       dimension=length(states), nnz=nnz(H), converged=true, matrix=H)
end

export N, PROTOTYPE_METADATA, PAULIS, wang_basis, multiply_words, heisenberg_dictionary
export FixedBlockChannel, block_isometry, fixed_block_channel, apply_channel, lift_coarse_word
export CertificateProblem, build_certificate, solve_certificate!, certificate_residual
export exact_diagonalization

end
