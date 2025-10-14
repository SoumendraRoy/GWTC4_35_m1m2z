import h5py
import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15 as lvk_cosmology
import astropy.units as u
import copy
from scipy.stats import truncnorm
import scipy.stats as ss
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
from tqdm import tqdm

seed = 1023895

################################################################################################################
## Load GWTC5 Selection File
def chi_eff_marginal(chi_eff, q):
    # Placeholder: replace with your actual marginalization function
    return np.ones_like(chi_eff)

def load_selection(file):
    with h5py.File(file, 'r') as obj:
        # Load dataset into DataFrame
        attrs = dict(obj.attrs.items())
        injections_no_cuts = obj['events'][:]
        
        # Implement SNR and FAR cuts: 
        # https://git.ligo.org/jaxen.godfrey/o4a-astro-dist-model-comparison-study/-/blob/main/analysis-scripts/inference.py?ref_type=heads#L180
        df = pd.DataFrame({key: injections_no_cuts[key][()] for key in injections_no_cuts.dtype.names})
        
        # Get metadata
        T_yr = attrs['total_analysis_time'][()] / (3600 * 24 * 365.25)
        N = attrs['total_generated'][()]

    # Compute new columns
    df['mass_1'] = df['mass1_source'] * (1 + df['redshift'])
    df['q'] = df['mass2_source'] / df['mass1_source']
    df['chi_eff'] = (df['spin1z'] + df['q'] * df['spin2z']) / (1 + df['q'])

    # Spin magnitudes
    a1 = np.sqrt(df['spin1x']**2 + df['spin1y']**2 + df['spin1z']**2)
    a2 = np.sqrt(df['spin2x']**2 + df['spin2y']**2 + df['spin2z']**2)

    # Spin sampling PDF
    spin_sampling_pdf = 1 / (16 * np.pi**2 * a1**2 * a2**2)

    # Sampling PDFs
    df['sampling_pdf_qchieff'] = (
        np.exp(df['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']) / spin_sampling_pdf * df['mass1_source'] *
        chi_eff_marginal(df['chi_eff'], df['q'])
    )

    df['sampling_pdf_q'] = np.exp(df['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']) * df['mass1_source']

    # Luminosity distance in Gpc (vectorized)
    df['luminosity_distance'] = lvk_cosmology.luminosity_distance(df['redshift'].values).to(u.Gpc).value
    
    # Comoving distance in Gpc
    dc = lvk_cosmology.comoving_transverse_distance(df['redshift'].values).to(u.Gpc).value
    
    # Hubble distance (c / H(z)) in Gpc
    dh_z = (2.99792e8 * u.m / u.s / lvk_cosmology.H(df['redshift'].values)).to(u.Gpc).value
    
    # Final PDF
    df['sampling_pdf_m1dqdlchieff'] = (
        df['sampling_pdf_qchieff'] / (1 + df['redshift']) / (dc + (1 + df['redshift']) * dh_z)
    )

    return df, T_yr, N

################################################################################################################

injection_file = "/mnt/home/sroy1/ceph/O4/O4a_final_selection/mixture-semi_o1_o2-real_o3_o4a/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf"
df, T_yr, N_selection = load_selection(injection_file)

# Modified from https://git.ligo.org/jaxen.godfrey/o4a-astro-dist-model-comparison-study/-/blob/main/analysis-scripts/inference.py?ref_type=heads#L180
fars = [df[key] for key in df.keys() if 'far' in key]
min_fars = np.min(fars, axis = 0)
snrs = df['semianalytic_observed_phase_maximized_snr_net']
found = (min_fars < 1) | (snrs > 10)
df = df[found]
print('Number of Selection Samples=', len(df), 'Dataframe Keys:', df.keys())

################################################################################################################

# Construction of Likelihood
with h5py.File('optimal_snr_aligo_O4low.h5', 'r') as inp:
    ms = np.array(inp['ms'])
    osnrs = np.array(inp['SNR'])

osnr_interp = RectBivariateSpline(ms, ms, osnrs)

def snr_unit_dl_unit_theta(m1z, m2z):
    return osnr_interp.ev(m1z, m2z)

m1z = np.array(df['mass_1'])
m2z = np.array(df['mass_1'])*np.array(df['q'])
dl = np.array(df['luminosity_distance'])
rho = np.array(df['estimated_optimal_snr_net'])
Theta = (1./3.**0.5)*dl*rho/snr_unit_dl_unit_theta(m1z, m2z)

print("Deleting the Theta>1 Samples:", len(np.where(Theta>1)[0])) # this can potentially give bias if lots of samples rejected.
m1z, m2z, dl, rho, Theta, df = m1z[Theta<1.], m2z[Theta<1.], dl[Theta<1.], rho[Theta<1.], Theta[Theta<1.], df[Theta<1.]
rho_obs = Theta*snr_unit_dl_unit_theta(m1z, m2z)/dl + np.random.randn(len(m1z))
print("Deleting the rho_obs<0 Samples:", len(np.where(rho_obs<0.)[0])) # this can potentially give bias if lots of samples rejected.
m1z, m2z, dl, rho, Theta, df, rho_obs = m1z[rho_obs>0.], m2z[rho_obs>0.], dl[rho_obs>0.], rho[rho_obs>0.], Theta[rho_obs>0.], df[rho_obs>0.], rho_obs[rho_obs>0.]

################################################################################################################

uncert = {
    'threshold_snr': 8,
    'Theta': 0.05,
    'mcz': 0.03,
    'logq': 0.15 # change it to 0.3 for cut in m1 to match the catalog errorbar of m1.
}

def mcz_add_err(Mcz, rho_obs, uncert, Nsamp):
    sigma_Mcz = uncert['threshold_snr']/rho_obs*uncert['mcz']
    logMczo = np.log(Mcz) + sigma_Mcz*np.random.randn(1)
    logMczs = logMczo + sigma_Mcz*np.random.randn(Nsamp)
    return np.exp(logMczs)

def logq_add_err(logq, rho_obs, uncert, Nsamp):
    sigma_logq = uncert['threshold_snr']/rho_obs*uncert['logq']
    logqo = truncnorm.rvs(a=-np.inf, b=-logq/sigma_logq, loc=logq, scale=sigma_logq)
    logqs = truncnorm.rvs(a=-np.inf, b=-logqo/sigma_logq, loc=logqo,
                          scale=sigma_logq, size=Nsamp)

    w = ss.norm.cdf(-logqo/sigma_logq)/ss.norm.cdf(-logqs/sigma_logq)
    logqs_reweighted = np.random.choice(logqs, size=Nsamp, p=w/np.sum(w), replace=True)

    return logqs_reweighted

def Theta_add_err(Theta, rho_obs, uncert, Nsamp):
    sigma_Theta = uncert['threshold_snr']/rho_obs*uncert['Theta']
    Thetao = truncnorm.rvs(a=-Theta/sigma_Theta, b=(1-Theta)/sigma_Theta, loc=Theta, scale=sigma_Theta)
    Thetas = truncnorm.rvs(a=-Thetao/sigma_Theta, b=(1-Thetao)/sigma_Theta, loc=Thetao,
                          scale=sigma_Theta, size=Nsamp)

    w = (ss.norm.cdf((1-Thetao)/sigma_Theta) - ss.norm.cdf(-Thetao/sigma_Theta))/(ss.norm.cdf((1-Thetas)/sigma_Theta) - ss.norm.cdf(-Thetas/sigma_Theta))
    Thetas_reweighted = np.random.choice(Thetas, size=Nsamp, p=w/np.sum(w), replace=True)
    
    return Thetas_reweighted

def rhos_samples(rho_obs, Nsamp):
    rhos = np.random.normal(rho_obs, 1, size=Nsamp)
    return rhos

def dl_add_err(dl, Mczs, logqs, Thetas, rhos, uncert, Nsamp):
    dfid = 1.
    m1zs = Mczs*(np.exp(logqs)**(-3./5.))*(1.+np.exp(logqs))**(1./5.)
    m2zs = Mczs*(np.exp(logqs)**(2./5.))*(1.+np.exp(logqs))**(1./5.)
    ds = snr_unit_dl_unit_theta(m1zs, m2zs)*Thetas/rhos
    
    return ds

################################################################################################################

Mcz = (m1z*m2z)**0.6/(m1z+m2z)**0.2
q = m2z/m1z
logq = np.log(q)
Nobs = Mcz.shape[0]
Nsamp = 10000

dfid = 1.

zinterp = np.linspace(0, 12, 100000)
dlinterp = lvk_cosmology.luminosity_distance(zinterp).to(u.Gpc).value

filename = "/mnt/ceph/users/sroy1/GWTC4/Selection_Samples_With_Mock_PE.h5"

with h5py.File(filename, "w") as h5f:
    df_info = h5f.create_group("info")
    df_info.create_dataset("analysis_time_yr", data=T_yr)
    df_info.create_dataset("total_injections", data=N_selection)

    # Save the original DataFrame as a group under "injections"
    df_group = h5f.create_group("injections")
    for col in df.columns:
        # Save each column in the "injections" group
        df_group.create_dataset(col, data=df[col].values)

    # Save the injections-pe as a group under "injections-pe"
    pe_group = h5f.create_group("injections-pe")
    
    for i in tqdm(range(Nobs)):
        Mczs = mcz_add_err(Mcz[i], rho_obs[i], uncert, Nsamp)
        logqs = logq_add_err(logq[i], rho_obs[i], uncert, Nsamp)
        Thetas = Theta_add_err(Theta[i], rho_obs[i], uncert, Nsamp)
        rhos = rhos_samples(rho_obs[i], Nsamp)
        ds = dl_add_err(dl[i], Mczs, logqs, Thetas, rhos, uncert, Nsamp)

        m1zs = Mczs*(np.exp(logqs)**(-3./5.))*(1.+np.exp(logqs))**(1./5.)
        m2zs = Mczs*(np.exp(logqs)**(2./5.))*(1.+np.exp(logqs))**(1./5.)
        qs = m2zs/m1zs
        etas = (m1zs*m2zs)/(m1zs+m2zs)**2.

        wfishbach = (m1zs-m2zs)*etas**0.6/(m1zs+m2zs)**2.
        wme = 1./((qs-1./qs)*etas**2.)
        w = wfishbach*wme*(Thetas*snr_unit_dl_unit_theta(m1zs, m2zs))/ds**2

        m1zs_reweighted = np.random.choice(m1zs, size=Nsamp, p=w/np.sum(w), replace=True)
        m2zs_reweighted = np.random.choice(m2zs, size=Nsamp, p=w/np.sum(w), replace=True)
        Thetas_reweighted = np.random.choice(Thetas, size=Nsamp, p=w/np.sum(w), replace=True)
        ds_reweighted = np.random.choice(ds, size=Nsamp, p=w/np.sum(w), replace=True)

        
        zs_reweighted = np.interp(ds_reweighted, dlinterp, zinterp)
        m1s_reweighted = m1zs_reweighted/(1.+zs_reweighted)
        m2s_reweighted = m2zs_reweighted/(1.+zs_reweighted)

        pe_group.create_dataset("Source_Frame_m1"+str(i), data=m1s_reweighted)
        pe_group.create_dataset("Source_Frame_m2"+str(i), data=m2s_reweighted)
        pe_group.create_dataset("Redshift"+str(i), data=zs_reweighted)
h5f.close()

print("Data saved to h5py file.")