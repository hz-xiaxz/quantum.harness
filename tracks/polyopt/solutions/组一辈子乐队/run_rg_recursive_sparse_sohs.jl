include("RGRecursiveSparseSOHS.jl")
using .RGRecursiveSparseSOHS
using LinearAlgebra

function print_result(result)
    println("variant=$(result["variant"])")
    println("  E_lb/N=$(result["E_lb_per_site"])")
    println("  PSD blocks=$(result["psd_block_dimensions"])")
    println("  real scalar variables=$(result["real_scalar_variables"])")
    println("  coefficient equalities=$(result["coefficient_equalities"])")
    println("  Mosek solve time=$(result["solve_time_seconds"]) s")
    println("  coefficient residual=$(result["coefficient_residual"])")
    flush(stdout)
end

println("setup metadata:")
for key in sort(collect(keys(PROTOTYPE_METADATA)))
    println("  $key=$(PROTOTYPE_METADATA[key])")
end
channel = fixed_block_channel()
println("channel: Φ(X)=JXJ†+Tr((I/2)X)Q, Q=I-JJ†")
println("  J size=$(size(channel.J)), rank(Q)=$(rank(channel.Q)), Kraus count=$(length(channel.kraus))")
println("  Kraus representation: K0=J; K_(a,s)=|q_a><s|/sqrt(2), a=1:6, s=up/down")
println("  completeness residual ||sum K K†-I||=$(norm(sum(K*K' for K in channel.kraus)-I))")
flush(stdout)

ed_elapsed = @elapsed ed = exact_diagonalization()
println("ED: E0/N=$(ed.energy_per_site), dimension=$(ed.dimension), nnz=$(ed.nnz), time=$ed_elapsed s")
flush(stdout)

baseline = build_certificate(rg=false)
base_result = solve_certificate!(baseline)
print_result(base_result)

augmented = build_certificate(rg=true)
rg_result = solve_certificate!(augmented)
print_result(rg_result)

println("comparison:")
println("  ED E0/N=$(ed.energy_per_site)")
println("  baseline E_lb/N=$(base_result["E_lb_per_site"])")
println("  RG E_lb/N=$(rg_result["E_lb_per_site"])")
println("  RG improvement/site=$(rg_result["E_lb_per_site"] - base_result["E_lb_per_site"])")
println("  inequalities baseline<=ED=$(base_result["E_lb"] <= ed.energy + 2e-7), RG<=ED=$(rg_result["E_lb"] <= ed.energy + 2e-7)")
flush(stdout)
