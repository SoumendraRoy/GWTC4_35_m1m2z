square(x) = x*x

raw"""
    log_dNdm(mbh, alpha, mtr, mbhmax, sigma)

Returns the log of the black hole mass function at `mbh` given parameters
describing the initial-final mass relation.

The black hole mass function here is derived by assuming a power law mass
function for the initial mass:

``\frac{\mathrm{d}N}{\mathrm{d}m} = \left( \frac{m}{m_\mathrm{max}}
\right)^{-\alpha}``

and that the average black hole mass is a piecewise function of the initial
mass, given by 

``m_\mathrm{BH}\left( m \right) = \begin{cases} m & m < m_\mathrm{tr} \\
    m_\mathrm{BH,max} - c \left( m - m_\mathrm{max} \right)^2 & m \geq m_\mathrm{tr}
\end{cases}``

This is a linear relationship for ``m < m_\mathrm{tr}``, and then smoothly
transitions to a quadratic with a maximum black hole mass of
``m_\mathrm{BH,max}`` achieved at ``m = m_\mathrm{max} = 2 m_\mathrm{BH,max} -
m_\mathrm{tr}`` (the smoothness condition sets ``c = 1/\left( 4 \left(
m_\mathrm{BH,max} - m_\mathhrm{tr}\right)\right)``).

Note that the function is normalized so that the power-law initial mass function
is unity at the initial mass that corresponds to the maximum expected black hole
mass.
"""
function log_dNdm(mbh, alpha, mtr, mbhmax, sigma)
    # mbh = mbhmax - c * (m - m_max)^2
    c = 1/(4*(mbhmax-mtr))
    m_max = 2*mbhmax - mtr

    if mbh < mbhmax
        a = mbhmax - mbh
        b = c
        x = a*a/(4*sigma*sigma)
        log_wt = log(sqrt(a*pi/(2*b))/(4*sigma)) + log(besselix(-0.25, x) + besselix(0.25, x))

        if mbh < mtr
            mlow = mbh
            mhigh = m_max + sqrt((mbhmax - mbh)/c)
            log_wt_low = 0.0
            log_wt_high = log_wt
        else
            d = sqrt((mbhmax - mbh)/c)
            mlow = m_max - d
            mhigh = m_max + d
            log_wt_low = log_wt
            log_wt_high = log_wt
        end
    else
        a = mbh-mbhmax
        b = c
        x = a*a/(4*sigma*sigma)

        log_wt = log(sqrt(a/(b*pi))/(4*sigma)) - 2*x + log(besselkx(0.25, x))

        mlow = m_max
        mhigh = m_max
        log_wt_low = log_wt
        log_wt_high = log_wt
    end

    logplow = -alpha*log(mlow / m_max)
    logphigh = -alpha*log(mhigh / m_max)

    logaddexp(log_wt_low + logplow, log_wt_high + logphigh)
end

function _c(mtr, mbhmax)
    1/(4*(mbhmax-mtr))
end

function _mini_max(mtr, mbhmax)
    2*mbhmax - mtr
end

function mrem_of_mini(mini, mtr, mbhmax)
    if mini < mtr
        return mini
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
    
        return mbhmax - c*(mini - m_max)^2
    end
end

function mini_left_of_mrem(mrem, mtr, mbhmax)
    if mrem > mbhmax
        return _mini_max(mtr, mbhmax)
    elseif mrem < mtr
        return mrem
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
        d = sqrt((mbhmax - mrem)/c)
    
        return m_max - d
    end
end

function mini_right_of_mrem(mrem, mtr, mbhmax)
    if mrem > mbhmax
        return _mini_max(mtr, mbhmax)
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
        d = sqrt((mbhmax - mrem)/c)
    
        return m_max + d
    end
end

function log_trapz(xs, log_ys)
    log_dx = log.(xs[2:end] .- xs[1:end-1])

    log_wts = log(0.5) .+ log_dx .+ logaddexp.(log_ys[1:end-1], log_ys[2:end])
    logsumexp(log_wts)
end

function mini_integral_log(mrem, alpha, mtr, mbhmax, sigma)
    m_max = _mini_max(mtr, mbhmax)

    mrem_low = max(0.01, mrem - 5*sigma) # Ensure positive
    mrem_high = mrem + 5*sigma

    if mrem_high < mbhmax
        # Then two distinct regions.

        mill = mini_left_of_mrem(mrem_low, mtr, mbhmax)
        milh = mini_left_of_mrem(mrem_high, mtr, mbhmax)
        mi = range(mill, stop=milh, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_i1 = log_trapz(mi, log_ys)

        mirl = mini_right_of_mrem(mrem_high, mtr, mbhmax)
        mirh = mini_right_of_mrem(mrem_low, mtr, mbhmax)
        mi = range(mirl, stop=mirh, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_i2 = log_trapz(mi, log_ys)

        logaddexp(log_i1, log_i2)
    elseif mrem_low < mbhmax
        # One regions
        ml = mini_left_of_mrem(mrem_low, mtr, mbhmax)
        mr = mini_right_of_mrem(mrem_low, mtr, mbhmax)
        mi = range(ml, stop=mr, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_trapz(mi, log_ys)
    else
        # Both are above.
        mr = max(0.01, mbhmax - 5*sigma)
        ml = mini_left_of_mrem(mr, mtr, mbhmax)
        mr = mini_right_of_mrem(mr, mtr, mbhmax)
        mi = range(ml, stop=mr, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_trapz(mi, log_ys)
    end
end

function make_log_dNdm_gridded_efficient(alpha, mtr, mbhmax, sigma; mmin=10.0, mmax=75.0)
    dm = 10*sigma / 128
    ms = collect(mmin:dm:mmax)

    log_dNdms = [mini_integral_log(m, alpha, mtr, mbhmax, sigma) for m in ms]

    function log_dNdm_gridded(m)
        interp1d(m, ms, log_dNdms)
    end
    log_dNdm_gridded
end

function make_log_dNdm_gridded(mgrid, alpha, mtr, mbhmax, sigma; mmin=10.0, mmax=100.0)
    ms = mgrid

    m_max = _mini_max(mtr, mbhmax)

    log_dNdms = [log_trapz(ms, [-alpha*log(mi / m_max) + logpdf(Normal(mrem_of_mini(mi, mtr, mbhmax), sigma), mr) for mi in ms]) for mr in ms]

    function log_dNdm_gridded(m)
        interp1d(m, ms, log_dNdms)
    end
    log_dNdm_gridded
end

function make_log_dNdm_peak_gridded(mgrid, mu_peak, sigma_peak, k3)
    ms = mgrid

    ztemp = (ms .- mu_peak) ./ sigma_peak
    H3 = ztemp .^ 3 - 3 .* ztemp
    log_dNdms = logpdf(Normal(mu_peak, sigma_peak), ms) + k3 .* H3

    function log_dNdm_gridded(m)
        interp1d(m, ms, log_dNdms)
    end
    log_dNdm_gridded
end

raw"""
    log_dNdq(q, beta)

The mass ratio distribution, given by 
``\frac{\mathrm{d}N}{\mathrm{d}q} = \left( \frac{1 + q}{2} \right)^\beta``

Note the normalization, where the value is `1` at `q = 1`.
"""
function log_dNdq(q, beta)
    beta*(log1p(q) - log(2))
end

raw"""
    log_mdsfr(z, lambda, zp, kappa)

Returns the log of the (un-normalized) Madau-Dickinson SFR:

``\frac{\left( 1 + z \right)^\lambda}{1 + \left(\frac{1 + z}{1 + z_p} \right)^\kappa}``
"""
function log_mdsfr(z, lambda, zp, kappa)
    lambda * log1p(z) - log1p(((1+z)/(1+zp))^kappa)
end

function make_log_dNdm_lm(log_rlm, m0, alphalm)
    function log_dN(m)
        log_rlm - alphalm*log(m/m0) - log1p(exp(m-m0)) + log(2)
    end

    log_dN
end

function make_log_dNdm_hm(log_r, m0, alpha)
    function log_dN(m)
        log_r - alpha*log(m/m0) - log1p(exp(-(m-m0))) + log(2)
    end

    log_dN
end

"""
Three–segment broken power-law log-pdf factory.

Segments (continuous at both breaks):
p(m) ∝ (m/mbreakf1)^(-alphalm1)                    for m ∈ [mlow, mbreakf1]
p(m) ∝ (m/mbreakf1)^(-alphamm1)                    for m ∈ (mbreakf1, mbreaks1]
p(m) ∝ (mbreaks1/mbreakf1)^(-alphamm1) * (m/mbreaks1)^(-alphahm1)  for m ∈ (mbreaks1, mhigh]

Normalization over [mlow, mhigh] is analytic; α = 1 cases handled.
"""
function make_combined_log_dNdm1(alphalm1, alphamm1, alphahm1, mbreakf1, mbreaks1; mlow=5.0, mhigh=300.0)
    @assert mlow < mbreakf1 < mbreaks1 < mhigh "Require mlow < mbreakf1 < mbreaks1 < mhigh."

    # helper: integral of (m/mref)^(-alpha) dm from a to b
    integrate_segment(a, b, alpha, mref) = (alpha == 1) ?
        (mref * log(b / a)) :
        (mref * ((b / mref)^(1 - alpha) - (a / mref)^(1 - alpha)) / (1 - alpha))

    # continuity factor at second break so that middle and high segments match at mbreaks1
    # middle at m=mbreaks1: (mbreaks1/mbreakf1)^(-alphamm1)
    # high uses (m/mbreaks1)^(-alphahm1), so multiply by C to equal the above at m=mbreaks1
    C = (mbreaks1 / mbreakf1)^(-alphamm1)
    logC = -alphamm1 * (log(mbreaks1) - log(mbreakf1))

    # normalization pieces
    Z1 = integrate_segment(mlow,      mbreakf1, alphalm1, mbreakf1)
    Z2 = integrate_segment(mbreakf1,  mbreaks1, alphamm1, mbreakf1)
    Z3 = C * integrate_segment(mbreaks1, mhigh, alphahm1, mbreaks1)

    Z = Z1 + Z2 + Z3
    @assert Z > 0 "Normalization constant is non-positive; check parameters."
    logZ = log(Z)

    lmbf1 = log(mbreakf1)
    lmbs1 = log(mbreaks1)

    function log_dN(m)
        if m < mlow || m > mhigh
            return -Inf
        elseif m <= mbreakf1
            return -alphalm1 * (log(m) - lmbf1) - logZ
        elseif m <= mbreaks1
            return -alphamm1 * (log(m) - lmbf1) - logZ
        else
            return logC - alphahm1 * (log(m) - lmbs1) - logZ
        end
    end

    return log_dN
end


function make_combined_log_dNdm(log_rlm, alphalm, log_rhm, alphahm, mu_peak, sigma_peak, k3; mgrid=collect(10:0.25:100), mnorm=35.0)
    log_dNdm = make_log_dNdm_peak_gridded(mgrid, mu_peak, sigma_peak, k3)
    log_dNdm_norm = log(mnorm) + log_dNdm(mnorm)

    log_rlm1 = log_rlm + log_dNdm(mu_peak) - log_dNdm_norm
    log_dNdm_lm = make_log_dNdm_lm(log_rlm1, mu_peak, alphalm)

    log_rhm1 = log_rhm + log_dNdm(mu_peak) - log_dNdm_norm
    log_dNdm_hm = make_log_dNdm_hm(log_rhm1, mu_peak, alphahm)

    function log_dN(m)
        logaddexp(logaddexp(log_dNdm(m) - log_dNdm_norm, log_dNdm_lm(m)), log_dNdm_hm(m))
    end

    log_dN
end

function make_log_dNdm1dqdVdt(alphalm1, alphamm1, alphahm1, mbreakf1, mbreaks1, log_rlm2, alphalm2, log_rhm2, alphahm2, mu_peak2, sigma_peak2, k32, beta, lambda, zp, kappa; mgrid=collect(10:0.25:100), mnorm=35.0, qnorm=1.0, znorm=0.0)
    log_dNdm_total1 = make_combined_log_dNdm1(alphalm1, alphamm1, alphahm1, mbreakf1, mbreaks1)
    log_dNdm_total2 = make_combined_log_dNdm(log_rlm2, alphalm2, log_rhm2, alphahm2, mu_peak2, sigma_peak2, k32; mgrid=mgrid, mnorm=mnorm)
    
    # These should be the values of the corresponding terms in the model at (mnorm, qnorm, znorm)
    log_znorm = log_mdsfr(znorm, lambda, zp, kappa)
    log_qnorm = log_dNdq(qnorm, beta)

    function log_dNdm1dqdVdt(m, q, z)
        m2 = q*m
        
        log_m1_pop = log_dNdm_total1(m)
        log_q_pop = log_dNdm_total2(m2) + log(m)
        log_pair_pop = log_dNdq(q, beta) - log_qnorm
        log_VT_pop = log_mdsfr(z, lambda, zp, kappa) - log_znorm
                
        log_m1_pop + log_q_pop + log_pair_pop + log_VT_pop
    end

    return log_dNdm1dqdVdt
end

function pe_dataframe_to_samples_array(df, Nposts; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    shuffled_evts = [shuffle(rng, evt) for evt in evts]
    pe_samples = [[[evt[i, :mass_1_source], evt[i, :mass_ratio], evt[i, :redshift]] for i in 1:np] for (np, evt) in zip(Nposts, shuffled_evts)]
    log_pe_wts = [vec(evt[1:np, :prior_logwt_m1qz]) for (np, evt) in zip(Nposts, shuffled_evts)]
    (pe_samples, log_pe_wts)
end

function evt_dataframe_to_kde(df, Nkde; rng=Random.default_rng())
    df = shuffle(rng, df)
    log_wts = .- li_nocosmo_prior_logwt_m1qz(df)
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:size(df, 1), Weights(wts), 2*Nkde)
    df_sel = df[inds, :]
    pts = Array(df_sel[1:2*Nkde, [:mass_1_source, :mass_ratio, :redshift]])'
    bw_opt_kde(pts[:, 1:Nkde], pts[:, Nkde+1:end])
end

function pe_dataframe_to_evt_kdes(df, Nkde; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    [evt_dataframe_to_kde(evt, Nkde; rng=rng) for evt in evts]
end

function sel_dataframe_to_samples_array(df, Nsamp=1024; rng=Random.default_rng())
    shuffled_df = shuffle(rng, df)
    sel_samples = [[shuffled_df[i, :mass1_source], shuffled_df[i, :q], shuffled_df[i, :redshift]] for i in 1:Nsamp]
    log_sel_pdraw = log.(shuffled_df[1:Nsamp, :sampling_pdf_q])
    (sel_samples, log_sel_pdraw)
end

@model function pop_model_samples(evt_samples, log_prior_wts, sel_samples, log_sel_pdraw, Ndraw, m_grid, zs_interp)
    nevt = length(evt_samples)
    dh = dH(h_lvk)
    dcs_interp = dc_over_dh(zs_interp, Ω_M_lvk)
    dvdz_interp = dvdz_over_vh(zs_interp, Ω_M_lvk, dcs_interp)

    log_dV_interp = 3*log(dh) .+ log.(dvdz_interp) .- log1p.(zs_interp)

    # Priors
    alphalm1 ~ Uniform(-50, 10)
    alphamm1 ~ Uniform(0, 30)
    alphahm1 ~ Uniform(0, 20)
    mbreakf1 ~ Uniform(20, 25)
    mbreaks1 ~ Uniform(30, 40)
    
    log_rlm2 ~ Uniform(log(0.01), log(0.5))
    alphalm2 ~ Uniform(-20, 5)

    log_rhm2 ~ Uniform(log(0.01), log(0.5))
    alphahm2 ~ Uniform(0, 6)

    mu_peak2 ~ Uniform(25, 45)
    sigma_peak2 ~ Uniform(1, 10)
    k32 ~ Uniform(-2, 1)

    beta ~ Uniform(-50, 10)

    lambda ~ Uniform(-10, 10)
    zp ~ Uniform(1, 4)
    kappa ~ Uniform(3, 8)

    # Pop density
    log_dNdm1dqdVdt = make_log_dNdm1dqdVdt(alphalm1, alphamm1, alphahm1, mbreakf1, mbreaks1, log_rlm2, alphalm2, log_rhm2, alphahm2, mu_peak2, sigma_peak2, k32, beta, lambda, zp, kappa; mgrid=m_grid)

    function log_pop_density(theta)
        m1, q, z = theta
        log_dNdm1dqdVdt(m1, q, z) + interp1d(z, zs_interp, log_dV_interp)
    end

    thetas = map(evt_samples) do samples
        map(samples) do s
            m1, q, z = s
            [m1, q, z]
        end
    end

    thetas_sel = map(sel_samples) do s
        m1, q, z = s
        [m1, q, z]
    end

    log_likelihood_sum, log_normalization_sum, model_genq = pop_model_body(log_pop_density, thetas, log_prior_wts, thetas_sel, log_sel_pdraw, Ndraw)
    Turing.@addlogprob! log_likelihood_sum
    Turing.@addlogprob! log_normalization_sum

    # Now we draw samples
    m1s = map(model_genq.thetas_popwt) do tp
        tp[1]
    end
    qs = map(model_genq.thetas_popwt) do tp
        tp[2]
    end
    zs = map(model_genq.thetas_popwt) do tp
        tp[3]
    end
    m2s = m1s .* qs

    m1_draw = model_genq.theta_draw[1]
    q_draw = model_genq.theta_draw[2]
    z_draw = model_genq.theta_draw[3]

    (Neff_sel = model_genq.Neff_sel, R = model_genq.R, Neff_samps = model_genq.Neff_samps, rlm2 = exp(log_rlm2), rhm2 = exp(log_rhm2), m1s = m1s, m2s = m2s, qs = qs, zs = zs, m1_draw = m1_draw, m2_draw = m1_draw * q_draw, q_draw = q_draw, z_draw = z_draw)
end

## Now for the cosmologically-varying version.
## Here the canonical variables are going to be: 
## Detector-frame m1, q, luminosity-distance.
## KDEs and draw probability will be in this space.
function pe_dataframe_to_cosmo_samples_array(df, Nposts; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    shuffled_evts = [shuffle(rng, evt) for evt in evts]
    pe_samples = [[[evt[i, :mass_1], evt[i, :mass_ratio], evt[i, :luminosity_distance]/1000] for i in 1:np] for (np, evt) in zip(Nposts, shuffled_evts)]
    log_pe_wts = [vec(evt[1:np, :prior_logwt_m1dqdl]) for (np, evt) in zip(Nposts, shuffled_evts)]
    (pe_samples, log_pe_wts)
end

function evt_dataframe_to_cosmo_kde(df, Nkde; rng=Random.default_rng())
    df = shuffle(rng, df)
    log_wts = .- li_nocosmo_prior_logwt_m1dqdl(df)
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:size(df, 1), Weights(wts), 2*Nkde)
    df_sel = df[inds, :]
    pts = Array(df_sel[1:2*Nkde, [:mass_1, :mass_ratio, :luminosity_distance]])'
    pts[4, :] ./= 1000 # Our distances are in Gpc
    bw_opt_kde(pts[:, 1:Nkde], pts[:, Nkde+1:end])
end

function pe_dataframe_to_cosmo_evt_kdes(df, Nkde; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    [evt_dataframe_to_cosmo_kde(evt, Nkde; rng=rng) for evt in evts]
end

function sel_dataframe_to_cosmo_samples_array(df, Nsamp=1024; rng=Random.default_rng())
    shuffled_df = shuffle(rng, df)
    sel_samples = [[shuffled_df[i, :mass_1], shuffled_df[i, :q], shuffled_df[i, :luminosity_distance]] for i in 1:Nsamp]
    log_sel_pdraw = log.(shuffled_df[1:Nsamp, :sampling_pdf_m1dqdl])
    (sel_samples, log_sel_pdraw)
end