const v0 = 0.01f0
const v1 = 0.99f0
const apd = 0.5f0

const NUM_STATES = 3

@begin_def 3

@def_state g_v        1
@def_state g_w        2
@def_state g_s        3

@end_def

H(x) = Atan(x*10000)/π + 0.5f0

function fenton_4v(T, n)
    E = eltype(T)
    S = zeros(E, (n, NUM_STATES))

    g_v(S, :, T(fill(E(1), (n,))))
    g_w(S, :, T(fill(E(1), (n,))))
    g_s(S, :, T(fill(E(0), (n,))))

    return S
end

create_model = fenton_4v

function update_currents(I0, u, S, i)
    tau_d = 0.065f0
    tau_si = 31.8364f0
    tau_0 = 39.0f0
    tau_a = 0.009f0
    u_c = 0.23f0
    u_v = 0.055f0
    # u_w = 0.146f0
    u_0 = 0.0f0
    u_m = 1.0f0
    u_so = 0.3f0
    a_so = 0.115f0
    b_so = 0.84f0
    c_so = 0.02f0

    v = g_v(S,i)
    w = g_w(S,i)
    s = g_s(S,i)

    I_fi = -v*H(u-u_c)*(u-u_c)*(u_m-u) / tau_d
    I_si = -w * s / tau_si

    tau_so = tau_0

    I_so = 0.5f0*(a_so-tau_a)*(1+Tanh((u-b_so)/c_so)) +
      (u-u_0)*(1-H(u-u_so))/tau_so +
      H(u-u_so)*tau_a

    return I0 + (I_fi + I_si + I_so)
end

function update_gates(u, S, i, Δt)
    u_c = 0.23f0
    u_w = 0.146f0
    tau_vp = 3.33f0
    tau_vn1 = 19.2f0
    tau_vn2 = 10.0f0
    tau_wp = 160.0f0
    tau_wn1 = 75.0f0
    tau_wn2 = 75.0f0
    r_sp = 0.02f0
    r_sn = 1.2f0
    k = 3
    u_csi = 0.8f0

    v = g_v(S,i)
    w = g_w(S,i)
    s = g_s(S,i)

    tau_vn = tau_vn1

    g_v(S, i,
        euler(
            v,
            (1-H(u-u_c))*(1-v)/tau_vn - H(u-u_c)*v/tau_vp,
            Δt
        )
    )

    tau_wn = tau_wn2*H(u-u_w) + tau_wn1*(1-H(u-u_w))

    g_w(S, i,
        euler(
            w,
            (1-H(u-u_c))*(1-w)/tau_wn - H(u-u_c)*w/(tau_wp*apd),
            Δt
        )
    )

    r_s = r_sp*H(u-u_c) + r_sn*(1-H(u-u_c))

    g_s(S, i,
        euler(
            s,
            r_s*(0.5f0*(1 + Tanh((u-u_csi)*k)) - s),
            Δt
        )
    )
end

function update_concentrations(u, S, i, Δt)
end
