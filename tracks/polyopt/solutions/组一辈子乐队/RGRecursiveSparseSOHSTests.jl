using Test
using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools

include("RGRecursiveSparseSOHS.jl")
using .RGRecursiveSparseSOHS

@testset "minimal RG-recursive sparse SOHS" begin
    channel = fixed_block_channel()
    I2 = PAULIS[1]
    I8 = Matrix{ComplexF64}(I, 8, 8)

    @testset "fixed unital CP channel" begin
        @test size(channel.J) == (8, 2)
        @test channel.J' * channel.J ≈ I2 atol=1e-12 rtol=0
        @test channel.Q ≈ I8 - channel.J * channel.J' atol=1e-12 rtol=0
        @test apply_channel(channel, I2) ≈ I8 atol=1e-12 rtol=0
        @test sum(K * K' for K in channel.kraus) ≈ I8 atol=2e-12 rtol=0
        X = ComplexF64[0.2 0.1+0.3im; 0.1-0.3im -0.4]
        @test sum(K * X * K' for K in channel.kraus) ≈ apply_channel(channel, X) atol=2e-12 rtol=0
    end

    @testset "sparse lifting identity" begin
        @test length(wang_basis(Val(12))) == 37
        @test length(wang_basis(Val(12); max_support=2)) == 145
        @test length(wang_basis(Val(4))) == 13
        lifted_identity = lift_coarse_word(channel, ntuple(_ -> UInt8(0), 4))
        @test length(lifted_identity) == 1
        @test only(values(lifted_identity)) ≈ 1.0 + 0im atol=1e-12
        coarse_x = ntuple(i -> UInt8(i == 1 ? 1 : 0), 4)
        lifted_x = lift_coarse_word(channel, coarse_x)
        local_reconstruction = zeros(ComplexF64, 8, 8)
        for (word, coefficient) in channel.pauli_images[2]
            local_reconstruction += coefficient * kron((PAULIS[Int(label)+1] for label in word)...)
        end
        @test local_reconstruction ≈ apply_channel(channel, PAULIS[2]) atol=2e-12 rtol=0
        @test all(word[4:12] == ntuple(_ -> UInt8(0), 9) for word in keys(lifted_x))
    end

    @testset "SOHS lifting, degeneration, and cone inclusion" begin
        baseline = build_certificate(rg=false)
        augmented = build_certificate(rg=true)
        @test baseline.metadata["psd_block_dimensions"] == [37]
        @test augmented.metadata["psd_block_dimensions"] == [37, 13]
        @test augmented.metadata["real_scalar_variables"] ==
              baseline.metadata["real_scalar_variables"] + 13^2
        @test augmented.metadata["baseline_included_verbatim"]
        @test augmented.metadata["rg_zero_face_is_baseline"]

        ed = exact_diagonalization()
        @test ed.dimension == 924
        @test issparse(ed.matrix)
        @test ed.converged
        base_result = solve_certificate!(baseline)
        rg_result = solve_certificate!(augmented)
        @test base_result["coefficient_residual"] < 2e-7
        @test rg_result["coefficient_residual"] < 2e-7
        @test rg_result["E_lb"] >= base_result["E_lb"] - 2e-7
        @test base_result["E_lb"] <= ed.energy + 2e-7
        @test rg_result["E_lb"] <= ed.energy + 2e-7
    end
end
