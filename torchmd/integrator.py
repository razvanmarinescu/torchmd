
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

def kinetic_energy(masses,vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=1,keepdim=True) * masses)
    return Ekin

def maxwell_boltzmann(masses,T):
    natoms = len(masses)
    return torch.sqrt(T*BOLTZMAN/masses)*torch.randn((natoms,3))  

def kinetic_to_temp(Ekin, natoms):
    return 2.0/(3.0 * natoms * BOLTZMAN) * Ekin 

def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel,gamma,coeff,dt,device):
    csi = torch.randn_like(vel).to(device)*coeff
    vel += -gamma*vel*dt + csi 

PICOSEC2TIMEU=1000.0/TIMEFACTOR
class Integrator:
    def __init__(self,systems,forces,timestep,device,gamma=None,T=None):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        gamma = gamma/PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            M=self.forces.par.masses
            self.vcoeff=torch.sqrt(2.0*gamma/M*BOLTZMAN*T*self.dt).to(device)


    def step(self, niter=1):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV(s.pos,s.vel,self.forces.forces, masses,self.dt)
            pot = self.forces.compute(s.pos,s.box)
            if self.T: langevin(s.vel,self.gamma,self.vcoeff,self.dt,self.device)
            _second_VV(s.vel, self.forces.forces, masses, self.dt)

        Ekin = kinetic_energy(masses,s.vel)
        T = kinetic_to_temp(Ekin, natoms)
        return Ekin,pot,T