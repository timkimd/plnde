function FHN(du,u,p,t) # FitzHugh-Nagumo oscillator
    du[1] = 30*u[1] - 10*u[1]^3 - 30*u[2]
    du[2] = 7.5*u[1]
end

function spiral(du,u,p,t) # Spiral
    true_A = [-0.1 -2.0 0; 2.0 -0.1 0; 0 0 -0.3]
    du .= true_A * (u .^ 3 + u)
end

function NDM(du,u,p,t) # Mutual Inhibition
    du[1] = 10 * (-u[1] + 1 ./ ( 1 + exp( 16 * ( u[2] - 0.5 ) ) ))
    du[2] = 10 * (-u[2] + 1 ./ ( 1 + exp( 16 * ( u[1] - 0.5 ) ) ))
end