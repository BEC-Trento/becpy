"""The module defines physical properties of atoms.

The base class Atom() specifies the properties that need to be filled in
for each particular atom, and functions to calculate properties like the
saturation intensity and recoil energy. It is meant to only be subclassed,
not used directly.

"""


from scipy.constants import physical_constants, h, hbar, pi, c as c0, atomic_mass as mp
aB = physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

class Atom(object):
    """Base class, not used directly. Defines a common interface."""

    name = None
    wavelength = None
    linewidth = None
    mass = None
    sclength = None

    @property
    def resonantfreq(self):
        """Frequency in Hz as determined from the resonant wavelength."""

        return c0/self.wavelength
    
    @property
    def cross_section(self):
        """ resonant scattering cross-section [m^2] """
        return 3*self.wavelength**2 / (2*pi)

    @property
    def sat_intensity(self):
        """ Saturation intensity [W/m^2] """
        return (pi*h*c0*self.linewidth) / (3*self.wavelength**3)
    
    @property
    def sat_photon_flux(self):
        return self.sat_intensity/(h*self.resonantfreq)
    
    @property
    def g_int(self):
        """ s-wave iInteraction constant """
        return 4*pi*hbar**2 * self.sclength / self.mass


    @property
    def Er(self):
        """Recoil energy for resonant wavelength of the atom."""

        return 2*hbar**2*pi**2/(self.mass*self.wavelength**2)

    @property
    def recoilfreq(self):
        """Recoil frequency in rad/s"""

        return self.Er/hbar



class Na23(Atom):
    """
    Properties of Sodium-23
    Attributes:
        name          : Sodium
        wavelength [m]: resonant on the D2 line
        sclength   [m]: |1, -1> pair collision
    """

    name = 'Sodium'
    wavelength = 589.162e-9 #D2 line
    linewidth = 2*pi*9.80e6
    mass = 23*mp
    sclength = 52*aB #|1, -1> pair collision

Na = Na23()