
# from pyne import data
from pyne import ace
from pyne import nucname
from pyne import nuc_data
from pyne.material_library import MaterialLibrary

from pyne.xs.data_source import SimpleDataSource
import pprint
import numpy as np
import os
import tables as tb

isos = SimpleDataSource()

IMPLEMENTED_REACTIONS = ["ElasticScattering",
                         "InelasticScattering",
                         "N2N",
                         "Fission"]

def get_punctual_cross_section(interaction_name, table):
    xs = None
    energy_in = None
    ang_cos = None
    ang_cdf = None
    if interaction_name == "ElasticScattering":
        scat = table.reactions[2]
        xs = scat.sigma
        energy_in = scat.ang_energy_in
        ang_cos = scat.ang_cos
        ang_cdf = scat.ang_cdf
        num_tabulated = len(energy_in) - 1
    elif interaction_name == "InelasticScattering":
        scat = table.reactions.get(4)
    elif interaction_name == "Fission":
        try:
            xs = table.reactions[18].sigma
        except:
            pass
    return xs, energy_in, ang_cos, ang_cdf


def get_multigroup_cross_section(interaction_name, nucid):
    xs = None
    if interaction_name == "ElasticScattering":
        xs = isos.reaction(nucid, 'scattering')
        pp = pprint.PrettyPrinter(indent=4)
    elif interaction_name == "Fission":
        try:
            xs = isos.reaction(nucid, 'fission')
        except:
            pass
    return xs

class Isotope:
    def get_cross_sections(self, i):
        return self.cross_sections[i]

class PunctualIsotope(Isotope):
    def __init__(self, mcnp_id):
        self.is_some = True
        self.mcnp_id = mcnp_id
        self.cross_sections = []
        self.is_fissile = False

        data_path = os.environ.get('CINTO_DATA')
        file_path = data_path+'/'+mcnp_id+".800nc"
        ace_file = ace.Library(file_path)
        try:
            ace_file.read()
        except:
            self.is_some = False
            return

        # print(ace_file.tables)
        table = ace_file.tables[mcnp_id+'.800nc']
        self.atomic_mass = table.awr

        # add energies
        self.energies = table.energy

        # add total cross section
        self.interactions = ["Total"]
        self.cross_sections.append(table.sigma_t)


        for interaction_name in IMPLEMENTED_REACTIONS:
            xs, energy_in, ang_cos, ang_cdf = get_punctual_cross_section(interaction_name, table)
            if xs is not None:
                self.interactions.append(interaction_name)
                self.cross_sections.append(xs)

        if "Fission" in self.interactions:
            self.is_fissile = True

class MultiGroupIsotope(Isotope):
    def __init__(self, nucid):
        self.mcnp_id = nucid
        self.cross_sections = []
        self.is_fissile = False
        self.atomic_mass = data.atomic_mass(nucid)
        self.is_some = True

        # add energies
        self.energies = np.asarray(isos.src_group_struct)

        # add total cross section
        self.interactions = ["Total"]
        self.cross_sections.append(isos.reaction(nucid, 'total'))

        for interaction_name in IMPLEMENTED_REACTIONS:
            xs = get_multigroup_cross_section(interaction_name, nucid)
            if xs is not None:
                self.interactions.append(interaction_name)
                self.cross_sections.append(xs)

        if len(self.interactions) == 1:
            self.is_some = False

        if "Fission" in self.interactions:
            self.is_fissile = True

class Material():
    def __init__(self,material_name):
        self.isotope_names = []
        self.concentrations = []
        self.num_isotopes = 0

        mats = MaterialLibrary(nuc_data, datapath="/materials")
        isotopes = mats[material_name]
        self.density = isotopes.density

        # loop on all isotopes in material
        for isotope in isotopes.comp:
            mcnp_isotope_index = str(isotope)[:-4]
            self.isotope_names.append(mcnp_isotope_index)
            self.concentrations.append(isotopes[isotope])
            self.num_isotopes+=1
        self.concentrations = np.asarray(self.concentrations)

    def get_isotope_names(self):
        return self.isotope_names

    def get_isotope(self, i):
        return self.isotopes[i]

    def get_num_isotopes(self):
        return len(self.isotopes)

    def get_concentrations(self):
        return self.concentrations

    def get_concentrations(self):
        return self.mcnp_id

def get_material(name):
    material = Material(name)
    return material

def get_isotope(name, evaluation_type="Punctual"):
    if evaluation_type == "Multigroup":
        return MultiGroupIsotope(name)
    return PunctualIsotope(name)

if __name__ =="__main__":

    material = Material("Concrete, Boron Frits-baryte")
    for isotope_name in material.isotope_names:
        isotope = get_isotope(isotope_name, evaluation_type="Punctual")
