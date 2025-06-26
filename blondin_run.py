# %% [markdown]
# ### Toy models (from Blondin+ 2023)

# %%
# general imports
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import radioactivedecay as rd

# TARDIS imports for the gamma-ray code
from tardis.io.atom_data import AtomData
from tardis.model import SimulationState
from tardis.plasma.base import BasePlasma
from tardis.io.configuration import config_reader
from tardis.energy_input.energy_source import get_nuclear_lines_database
from tardis.energy_input.gamma_ray_transport import get_taus
from tardis.energy_input.main_gamma_ray_loop import run_gamma_ray_loop, get_effective_time_array
from tardis.energy_input.gamma_ray_channel import create_isotope_dicts, create_inventories_dict, \
            calculate_total_decays, create_isotope_decay_df, \
            time_evolve_cumulative_decay

#%config InlineBackend.figure_format = 'retina'


# %%
# Download the atom data file from tardis-refdata repo to run this cell.
atom_data_file = '/home/duttaan2/Downloads/tardis-data/kurucz_cd23_chianti_H_He.h5'
atom_data = AtomData.from_hdf(atom_data_file)


# %%
# # This is the time start of the simulation
# # The mass fractions are decayed to this time

config = config_reader.Configuration.from_yaml("/home/duttaan2/Projects/gamma_ray_paper/config/tardis_config_merger_2012.yml")
config.supernova.time_explosion = 2.0 * u.day


# %%
# Create the model
# Model has been decayed to 2 days from model_isotope_time
model = SimulationState.from_csvy(config, atom_data)


# %%
time_start = 2.0
time_end = 100.0
time_steps = 500
time_space = 'log'
seed = 1
positronium_fraction = 1.0
path_to_decay_data = atom_data_file

# %%
# in days
times, effective_time_array = get_effective_time_array(time_start, time_end, time_space, time_steps)  

# %%
# Load the nuclear data
gamma_ray_lines = get_nuclear_lines_database(atom_data_file)
# We are using isotope abundance at the start of the simulation
raw_isotope_abundance = model.composition.isotopic_mass_fraction
#isotopic_mass_fraction = model.composition.
# Calculate the shell masses
shell_masses = model.volume * model.density

#initial_decay_energy = calculate_initial_decay_energy(raw_isotope_abundance, shell_masses, gamma_ray_lines, time_start)
# Create the isotope dictionary and inventories dictionary 
isotope_dict = create_isotope_dicts(raw_isotope_abundance, shell_masses)
inventories_dict = create_inventories_dict(isotope_dict)

# Calculate the total decays
total_decays = calculate_total_decays(inventories_dict, time_end - time_start)
# Create the isotope decay dataframe (This contains the decay information of all isotopes for all time steps)
isotope_decay_df = create_isotope_decay_df(total_decays, gamma_ray_lines)

# Time evolve the mass fraction
#time_evolved_mass_fraction = time_evolve_mass_fraction(raw_isotope_abundance, times)

# Get the taus and parents
taus, parents = get_taus(raw_isotope_abundance)

# Time evolved decay data frame
time_evolved_cumulative_decay_df = time_evolve_cumulative_decay(raw_isotope_abundance, shell_masses, gamma_ray_lines, times)

# %%
escape_energy, escape_energy_cosi, packets_df_escaped, gamma_ray_deposited_energy, total_deposited_energy, escape_luminosity = run_gamma_ray_loop(model,
    isotope_decay_df,
    time_evolved_cumulative_decay_df,
    50000000,
    times,
    effective_time_array,
    seed,
    positronium_fraction=positronium_fraction,
    spectrum_bins=500,
    grey_opacity=-1)

# %%
#time_index = escape_energy_cosi.columns.shape[0] - 1
#spectra_df = pd.DataFrame({'energy_keV': escape_energy_cosi.index, 'lum_density': escape_energy_cosi.iloc[:,time_index]})

# %%
#plt.plot(spectra_df['energy_keV'], spectra_df['lum_density'], 'r-')
#plt.loglog()
#plt.xlabel('Energy (keV)')
#plt.ylabel('Luminosity Density (erg/s/keV)')

# %%
edep_tardis = total_deposited_energy.T.sum(axis=1)
t_tardis = edep_tardis.index.values / 86400.0

# %%


# %%
# The 0.03 is the contribution from positrons.
# 0.6 is the mass of Ni-56

def analytic_estimate(t):
    return 0.678 * (0.97 * (1 - np.exp(-(40 / t)**2)) + 0.03) * (6.45 * np.exp(-t/8.8) + 1.45 * np.exp(-t/111.3)) * 1e43

# %%
#plt.plot(t_tardis, edep_tardis / analytic_estimate(t_tardis), "-", label="tardis")
#plt.xlim(0, 80)

#plt.ylim(0.8, 1.2)
#plt.xlim(2, 100)



# %%
# Save the PLOT DATA

edep_analytic_dep_df = pd.DataFrame({'time': t_tardis, 'edep_analytic': analytic_estimate(t_tardis), 'edep_tardis': edep_tardis})

edep_analytic_dep_df.to_csv('/home/duttaan2/Projects/gamma_ray_paper/results/merger_2012_100d_500ts_1e9_log_ps1_edep_kev.csv', header=None, index=False)
escape_energy.to_csv('/home/duttaan2/Projects/gamma_ray_paper/results/merger_2012_100d_500ts_1e9_log_ps1_escape_energy_kev.csv', header=True, index=True)

# %%
#plt.plot(np.geomspace(2, 100, 500))

# %%



