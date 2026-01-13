starttime = time()
#### replicating den toom
using Plots
using LinearAlgebra
using BifurcationKit
using Statistics
using NLsolve
using JLD2
using DifferentialEquations
timeafterpackages = time() - starttime
println("Time after loading packages: $timeafterpackages seconds")

#save data tussendoor
tussenstap = 100000

#gridsize
n_z = 10
z = range(-1, 0, length=n_z)
delta_z = 1/(n_z-1)

#parameters
i_res_T = 1 
i_res_S = 0 
F_0 = 0* 100 #10^2 #turn of convection
epsilon = 10
P = 10^3

i_res_plus = (i_res_S + i_res_T) / 2
i_res_minus = (i_res_S - i_res_T) / 2

#bif parameter: freshwater forcing
gamma = -1.0

rho_ref = 0.0

function second_order_derivative_matrix(n)
    d = -2  # Main diagonal value
    offd = 1  # Off-diagonal value
    M = diagm(-1 => ones(n-1) * offd, 0 => ones(n) * d, 1 => ones(n-1) * offd) # Create the tridiagonal matrix
    #no flux at the boundaries
    M[1,2] = 2 
    M[n,n-1] = 2
    return M
end

#forcing terms
function forcing_terms(gamma)
    T_tilde = 0 .* - cos.(2 .* pi .* z) #T reversed, to be more similar to a realistic temperature profile # TURNED OFF
    S_tilde = gamma .* cos.(π .* z)
    rho_tilde = S_tilde - T_tilde
    mu_tilde = S_tilde + T_tilde
    return rho_tilde, mu_tilde
end

function conv(d_rho)
    #smooth transition (G)
    # return (F_0 / (2*P))*(1 + tanh.(epsilon * (d_rho.-rho_ref)))

    #approx step function (F)
    return (F_0/P) .* max.(0.0, tanh.((epsilon * d_rho).^3))
end

function nonlinear(rho,mu)
    #note that now i=1 corresponds to the bottom box
    #and i=numb_boxes corresponds to the top box
    nonlinear_rho = similar(rho)
    nonlinear_mu = similar(mu)

    for i in 1:n_z
        if i == 1
            mu_fwd = (mu[i+1] -  mu[i])/delta_z
            rho_fwd = (rho[i+1] -  rho[i])/delta_z
            coeff = conv(rho_fwd)

            nonlinear_rho[i] = coeff * rho_fwd /delta_z
            nonlinear_mu[i] = coeff * mu_fwd /delta_z
        elseif i == n_z 
            mu_bwd = (mu[i] - mu[i-1])/delta_z
            rho_bwd = (rho[i] - rho[i-1])/delta_z
            coeff = conv(rho_bwd)

            nonlinear_rho[i] = - coeff * rho_bwd /delta_z
            nonlinear_mu[i] = - coeff * mu_bwd / delta_z
        else
            mu_fwd = (mu[i+1] -  mu[i])/delta_z
            mu_bwd = (mu[i] - mu[i-1])/delta_z
            rho_fwd = (rho[i+1] -  rho[i])/delta_z
            rho_bwd = (rho[i] - rho[i-1])/delta_z
            coeff_fwd = conv(rho_fwd)
            coeff_bwd = conv(rho_bwd)

            nonlinear_rho[i] = ((coeff_fwd * rho_fwd) - (coeff_bwd * rho_bwd))/delta_z
            nonlinear_mu[i] = ((coeff_fwd * mu_fwd) - (coeff_bwd * mu_bwd))/delta_z
        end
    end
    return nonlinear_rho, nonlinear_mu
end

A_linear = second_order_derivative_matrix(n_z)


function d_dt(rho_mu, p)
    gamma = p[1]
    rho = rho_mu[1:n_z]
    mu  = rho_mu[n_z+1:end]

    
    #forcing terms
    rho_tilde, mu_tilde = forcing_terms(gamma)
    #linear operator
    nonlinear_rho, nonlinear_mu = nonlinear(rho, mu)
    
    #returning the system of equations
    drho_dt = (1/(delta_z^2 *P))* A_linear * rho - i_res_plus * rho - i_res_minus * mu + rho_tilde + nonlinear_rho
    dmu_dt = (1/(delta_z^2 *P))* A_linear * mu - i_res_plus * mu - i_res_minus * rho + mu_tilde + nonlinear_mu

    return vcat(drho_dt, dmu_dt)
end


function d_dt_constrained(rho_mu_λ, p)
    rho_mu = rho_mu_λ[1:end-1]
    λ = rho_mu_λ[end]

    F = d_dt(rho_mu, p)

    # Compute ∇h · λ and add it to F
    constraint_grad = ones(2n_z) .* 0.5  # d(∫S dz)/dx
    F_with_constraint = F .+ λ .* constraint_grad

    # Constraint equation
    S = (rho_mu[1:n_z] .+ rho_mu[n_z+1:2n_z]) ./ 2
    constraint = sum(S) * delta_z

    return vcat(F_with_constraint, constraint)
end


function find_equilibrium(gamma; tol=1e-5)
    scaling_vec = range(0, 1, length=n_z)
    rho_tilde, mu_tilde = forcing_terms(gamma)
    #rho_mu_init = vcat(rho_tilde, mu_tilde)
    rho_mu_init = vcat(rho_tilde.*scaling_vec, mu_tilde.*scaling_vec)

    # rho_mu_init = 0.5*ones(n_z*2)
    # rho_mu_init = zeros(n_z*2)

    # Define root-finding problem: solve d_dt(delta_rho, [f]) = 0
    function equilibrium_eq!(residual, rho_mu)
        residual .= d_dt(rho_mu, [gamma])  # Compute d/dt
    end
    # Solve the system using nlsolve
    sol = nlsolve(equilibrium_eq!, rho_mu_init, ftol=tol, method=:newton)

    if sol.f_converged
        println("Equilibrium found for gamma = $gamma")
    else
        error("No convergence for gamma = $gamma")
    end

    # REMOVE THE CONSTANT, such that if removed ∫S dz = 0
    extra_const_S = sum((sol.zero[1:n_z] .+ sol.zero[n_z+1:2n_z])/2)/ n_z
    sol.zero[1:n_z] .-= extra_const_S
    sol.zero[n_z+1:2*n_z] .-= extra_const_S

    return sol.zero
end

equilibriumstate = find_equilibrium(gamma)

########################## PLOT EQUILIBRIUM STATE ##########################
# rho = equilibriumstate[1:n_z]
# mu = equilibriumstate[n_z+1:2n_z]

# # Compute S and T
# S = (rho .+ mu) ./ 2
# T = (mu .- rho) ./ 2

# # Plotting
# plot(rho, z, label = "ρ", xlabel = "Value", ylabel = "z", lw=2, title = "Solution at γ = $(round(gamma; digits=4))")
# plot!(S, z, label = "S", lw=2, linestyle = :dot)
# plot!(T, z, label = "T", lw=1, linestyle = :dash, size=(600, 800))
# xlims!(-2, 2)
# ylims!(-1, 0)

# savefig("extract_parameters/dentoom_nosalt.png")

########################## TIME EVOLUTION ###########################
temp = -3 .* ones(n_z)
sal = 0 .* equilibriumstate[1:n_z]
rho_mu = vcat((sal .- temp), (sal.+temp)) #start from Temp = 0 degrees, salanity = 35 psu
deltat = 10*3600*24
nt = 150*365*24*3600//deltat
lsave = 50*3600*24//deltat
p = gamma
times = []
temperatures = []
salinities = []

anim = Animation()
#anim = @Plots.animate for t in 1:nt
 for t in 1:nt
    rho_mu .+= (deltat/(5*365*24*3600)) .* d_dt(rho_mu, p) #got to account for timescale

    if t % lsave == 0
        rho = rho_mu[1:n_z]
        mu  = rho_mu[n_z+1:2n_z]
        S   = (rho .+ mu) ./ 2
        T   = (mu .- rho) ./ 2
        T_dim = (T.*5) .+ 15 #dimensionalize temperature
        S_dim = (S.*(5/7.6)) .+ 35 #dimensionalize salinity
        push!(times, t*deltat/3600)
        push!(temperatures, T_dim[5])
        push!(salinities, S_dim[end]) #top salinity
        
        # Plots.plot(T_dim, z.*4000, label="T", lw=1, linestyle=:dash)
        # title!("Time = $(round(t*deltat/3600; digits=1)) h")
        # ylims!(-4000, 0)
        # xlims!(0, 20)

        Plots.plot(S_dim, z.*4000, label="S", lw=1, linestyle=:dash)
        title!("Time = $(round(t*deltat/3600; digits=1)) h")
        ylims!(-4000, 0)
        xlims!(0, 70)
        frame(anim)
    end
end

gif(anim, "extract_parameters/dentoom_salinityflux.gif")

plot(times, salinities, xlabel="Time (h)", ylabel="Salinity at top (°C)", lw=2, title="Salinity flux at top")
savefig("extract_parameters/dentoom_salinityflux_top.png")

########################## BIFURCATION ANALYSIS ##########################
# global i = 0
# prob = BifurcationProblem(d_dt_constrained,
#         vcat(equilibriumstate, 0.0),
#         [gamma], # set of parameters
#         IndexLens(1),     # parameter index for continuation
#         # PLOT THE BIFURCATION DIAGRAM
#         # record_from_solution = (rho_mu_lambda,p; k...) -> mean(rho_mu_lambda[1:end-1])) #y axis: norm x (delta_rho)
#         # SAVE THE SOLUTIONS
#         #record_from_solution = (d_rho,p; k...) -> deepcopy(d_rho))
#         #PLOT THE BIFURCATION DIAGRAM AS IN DEN TOOM: y axis = sum F
#         record_from_solution = (rho_mu, p; k...) -> 
#         begin
#             global i += 1
#             if i%tussenstap == 0
#                 println("Step $i: i saved the data")
#                 @save "save_solution.jld2" rho_mu = deepcopy(rho_mu) param = deepcopy(p)
#                 #@save "dentoom_replicating_nz$(n_z)_rhoref$(rho_ref)_start$(gamma).jld2" x = br.x params = br.param ds = br.ds n_unstable = br.n_unstable eig = br.eig specialpoints=br.specialpoint
#                 println("Saved solution for parameter $p")
#                 GC.gc() # reduce memoroy
#             end

#             rho = rho_mu[1:n_z]
#             sum_conv = 0.0
#             for i in 1:n_z-1
#                 delta_rho = (rho[i+1] - rho[i]) / delta_z
#                 sum_conv += (P/F_0)*conv(delta_rho)
#             end
#             return sum_conv
            
#         end)

#         contParams = ContinuationPar(
#             p_min = -0.3, p_max = 1.0, #-2.0  # Parameter range
#     dsmax = 0.000001, #0.02
#     ds = -0.000001, #start at -2 and go to -6
#     dsmin = 0.00000001,
#     max_steps = 100000000,              # Increase max steps
#     detect_bifurcation = 3,       # Detect bifurcations (higher means stricter)
#     tol_stability = 1e-8      # Improve stability tolerance
#     ,nev=2
#     , newton_options=NewtonPar(tol=1e-5, max_iterations=100))

# ### continue the branch (replica)
# br = continuation(prob, PALC(), contParams)
# show(br) 

# @save "dentoom_replicating_nz$(n_z)_rhoref$(rho_ref)_start$(gamma).jld2" x = br.x params = br.param ds = br.ds n_unstable = br.n_unstable eig = br.eig specialpoints=br.specialpoint

timecode = time() - starttime
println("Total time taken: $timecode seconds")