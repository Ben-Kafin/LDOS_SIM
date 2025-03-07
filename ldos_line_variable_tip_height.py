from numpy import array,dot,exp,zeros
from numpy.linalg import norm
import numpy as np
import sys
import matplotlib.pyplot as plt
import getopt
from os.path import exists,getsize
from os import getcwd,chdir
from time import time
from pathos.multiprocessing import ProcessPool
from lib import parse_doscar,parse_poscar,tunneling_factor
from scipy.ndimage import gaussian_filter
import copy
from scipy.integrate import simpson

#EXAMPLE CONSOLE INPUT FOR LDOS LINE SIM
#test = ldos_line('C:/Users/Benjamin/Documents/VASP/NHC/iPr/SAM/NHC2Au_complexes/p2/spinorb/freegold1/freegold2/freegold3/kpoints551')
#test.parse_VASP_output()
#test.calculate_ldos(64, 0.21, 0.19,np.array([-1,-1,0]),np.array([0.77193,0.71930,0]), tip_disp=15, unit_cell_num=1, phi=5.25)
#test.plot_map(cmap='vivid',show_colorbar=True)
#ldos_data=test.plot_energy_slice(0.2)
#test1 = ldos_line('C:/Users/Benjamin/Documents/VASP/NHC/iPr/SAM/NHC2Au_complexes/p2/spinorb/freegold1/freegold2/freegold3/kpoints551')
#test1.parse_VASP_output()
#test1.calculate_ldos(64, 1.1, -2.2,np.array([-1,-1,0]),np.array([0.77193,0.71930,0]), tip_scaler=1, ldos_data=ldos_data, tip_dist=7.5,unit_cell_num=1, phi=5.25)
#test1.normalize_position_slices(norm_range='full')
#test1.plot_map(cmap='vivid',show_colorbar=True)


class ldos_line:
    def __init__(self,filepath):
        self.npts=64
        self.emax=0
        self.emin=0
        self.estart=0
        self.eend=0
        self.x=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.y=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.z=array([[0.0 for i in range(self.npts)] for j in range(self.npts)])
        self.ldos=array([[0.0 for j in range(self.npts)] for i in range(self.npts)])
        self.exclude_args=['none']
        self.exclude=[]
        self.plot_atoms=[]
        self.nprocs=1
        self.periodic_coord=[]
        self.tip_disp=15.0
        self.tip_dist=15.0
        self.unit_cell_num=4
        self.position_slices=0
        self.energy_slices=0
        self.energy_sums=0
        self.esum_fig = None 
        self.esum_ax = None
        self.energies=[]
        self.orbitals=[]
        
        chdir(filepath)
    
    #reads in the POSCAR and DOSCAR files
    def parse_VASP_output(self,**args):
        if 'doscar' in args:
            doscar=args['doscar']
        else:
            doscar='./DOSCAR'
            
        if 'poscar' in args:
            poscar=args['poscar']
        elif exists('./CONTCAR'):
            if getsize('./CONTCAR')>0:
                poscar='./CONTCAR'
            else:
                poscar='./POSCAR'
        else:
            poscar='./POSCAR'
                
        try:
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar)[:4]
            #uncomment the next two lines to set all adlayer atom heights equal - TESTING
            #for i in range(-300,0):
            #    self.coord[i,2]=np.max(self.coord[:,2])
            self.dos, self.energies, self.ef, self.orbitals = parse_doscar(doscar)
            #uncomment the next line to set LDOS equal for all atoms - for TESTING
            #self.dos=np.ones((859,3,5000))
            #for i in range(3):
            #    self.dos[570,i]*=1000000
        except:
            print('error reading input files')
            sys.exit()
            
    def set_path_from_atoms(self,path_atoms,**args):
        path=[]
        for i in path_atoms:
            if len(i)>1:
                tempvar=i[1:]
            else:
                tempvar=[0,0]
            path.append(copy.deepcopy(self.coord[i[0]-1,:2]))
            for j in range(2):
                path[-1]+=self.lv[j,:2]*float(tempvar[j])
                
        #adds tolerance to the initial and final positions specified by the path
        idiff=(path[1]-path[0])/np.linalg.norm(path[1]-path[0])
        counter=0
        while True in [np.isnan(i) for i in idiff]:
            idiff=(path[counter]-path[0])/np.linalg.norm(path[counter]-path[0])
            counter+=1
        fdiff=(path[-1]-path[-2])/np.linalg.norm(path[-1]-path[-2])
        counter=0
        while True in [np.isnan(i) for i in fdiff]:
            fdiff=(path[-1]-path[counter])/np.linalg.norm(path[-1]-path[counter])
            counter+=1
            
        path_length=sum([np.linalg.norm(path[i]-path[i-1]) for i in range(1,len(path))])
        
        step_points=np.zeros(len(path)-1,dtype=np.int8)
        for i in range(1,len(path)):
            step_points[i-1]=round(np.linalg.norm(path[i]-path[i-1])/path_length*self.npts)-1
            if step_points[i-1]==1:
                step_points[i-1]+=1
        step_points[0]+=1
        self.npts=int(sum(step_points))
        path_distance=np.array([path_length*i/(self.npts-1) for i in range(self.npts)])
        path_coord=[path[0]]
        for i in range(1,len(path)):
            for j in range(step_points[i-1]):
                if i==1 and j==0:
                    pass
                else:
                    path_coord.append(path[i-1]+(path[i]-path[i-1])/(step_points[i-1]-1)*j)
        path_coord=np.array(path_coord)
        
        for i in range(len(path_coord)):
            path_coord[i]=np.dot(path_coord[i],np.linalg.inv(self.lv[:2,:2]))
            for j in range(2):
                while path_coord[i][j]>=1.0 or path_coord[i][j]<0.0:
                    if path_coord[i][j]>=1.0:
                        path_coord[i][j]-=1.0
                    if path_coord[i][j]<0.0:
                        path_coord[i][j]+=1.0
            path_coord[i]=np.dot(path_coord[i],self.lv[:2,:2])
                
        return path_coord,path_distance
            
    #sets the color and size of atoms overlayed on the topography
    #by default, all projected atoms are black and equally sized
    def set_atom_appearance(self,colors,sizes):
        self.atom_colors=[]
        self.atom_sizes=[]
        for i in range(len(self.atomtypes)):
            self.atom_colors.append(colors[i])
            self.atom_sizes.append(sizes[i])
            
    def plot_path(self,size=1):
        self.path_fig,self.path_ax=plt.subplots(1,1,tight_layout=True)
        tempx=[]
        tempy=[]
        colors=[]
        sizes=[]
        #plots the overlaid atoms as a scatterplot
        for i in range(len(self.coord)):
            for k in range(-1*size,size+1):
                for l in range(-1*size,size+1):
                    for j in range(len(self.atomtypes)):
                        if i < sum(self.atomnums[:j+1]):
                            break
                    tempx.append(self.coord[i][0]+self.lv[0,0]*k+self.lv[1,0]*l)
                    tempy.append(self.coord[i][1]+self.lv[0,1]*k+self.lv[1,1]*l)
                    sizes.append(self.atom_sizes[j])
                    colors.append(self.atom_colors[j])
                        
        atom_scatter=self.path_ax.scatter(tempx,tempy,color=colors,s=sizes)
        self.path_ax.plot(self.x,self.y,color='black',lw=5)
        self.path_ax.set(xlabel=r'position / $\AA$', ylabel=r'position / $\AA$')
        self.path_ax.set_aspect('equal')
        self.path_fig.show()
            
    #the ldos line is written to a file in the current directory with the following format:
    #3 lines of informational header
    #1 blank line
    #self.npts lines containing the positional values
    #1 blank line
    #self.npts lines containing the energy values
    #1 blank line
    #len(self.orbitals) sections each containing self.npts lines, each containing self.eend-self.estart ldos values
    def write_ldos(self):
        filename='./line_E{}to{}V_D{}_X{}_N{}_W{}_U{}'.format(self.emin,self.emax,self.tip_disp,','.join(self.exclude_args),self.npts,self.phi,self.unit_cell_num)
        with open(filename, 'w') as file:
            file.write('integration performed from {} to {} V over {} energy points\n'.format(self.emin,self.emax,self.eend-self.estart))
            file.write('atoms types excluded from DOS integration: ')
            for i in self.exclude_args:
                file.write('{} '.format(i))
            file.write('\norbital contributions to ldos: {}'.format(', '.join(self.orbitals)))
            file.write('\n\n')
            for axes,k in zip([self.path_distance,self.energies],range(2)):
                for i in range(self.npts):
                    for j in range(self.eend-self.estart):
                        if k==0:
                            file.write(str(axes[i]))
                        else:
                            file.write(str(axes[j]))
                        file.write(' ')
                    file.write('\n')
                file.write('\n')
            for projection in self.ldos:
                for i in range(self.npts):
                    for j in range(self.eend-self.estart):
                        file.write(str(projection[i][j]))
                        file.write(' ')
                    file.write('\n')
                file.write('\n')
                
    #reads in the ldos file created by self.write_ldos()
    def parse_ldos(self,filepath):
        header=filepath.split('_')
        if header[0][-3:]!='line':
            print('not a ldos line file. exiting...')
            sys.exit()
        
        erange=header[1][1:-1].split('to')
        self.emin=float(erange[0])
        self.emax=float(erange[1])
        self.tip_disp=float(header[2][1:])
        self.exclude=header[3][1:].split(',')
        self.npts=int(header[4][1:])
        self.phi=float(header[5][1:])
        self.unit_cell_num=int(header[6][1:])
        
        with open(filepath,'r') as file:
            lines=file.readlines()
            nedos=int(lines[0].split()[8])
            self.orbitals=lines[2].split(', ')
            self.path_distance=array([[0.0 for i in range(nedos)] for j in range(self.npts)])
            self.energies=array([[0.0 for i in range(nedos)] for j in range(self.npts)])
            self.ldos=array([[[0.0 for j in range(nedos)] for i in range(self.npts)] for k in range(len(self.orbitals))])
            for i in range(self.npts):
                for j in range(nedos):
                    self.path_distance[i][j]=lines[4+i].split()[j]
                    self.energies[i][j]=lines[5+self.npts+i].split()[j]
                    for k in range(len(self.orbitals)):
                        self.ldos[k][i][j]=lines[6+k+(2+k)*self.npts+i].split()[j]
        
    def reference_ldos(self,ref_filepath):
        ref=ldos_line(ref_filepath)
        ref.parse_VASP_output()
        ref_emax=ref.energies[self.eend]-ref.energies[self.estart]+self.emin
        ref.calculate_ldos(self.npts,ref_emax,self.emin,self.lv_path,self.lv_origin,phi=self.phi,unit_cell_num=unit_cell_num,exclude=self.exclude)
        self.ldos-=ref.ldos
    
    def calculate_ldos(self, npts, emax, emin, lv_path, lv_origin, **args):
        self.emax = emax
        self.emin = emin
        self.npts = npts
        self.lv_path = np.dot(np.array(lv_path), self.lv)
        self.lv_origin = np.dot(np.array(lv_origin), self.lv)
        self.x = np.array([0.0 for i in range(self.npts)])
        self.y = np.array([0.0 for i in range(self.npts)])
        self.z = np.array([0.0 for i in range(self.npts)])
        
        if 'path_atoms' in args:
            self.path_coord, self.path_distance = self.set_path_from_atoms(args['path_atoms'])
        else:
            self.path_distance = np.array([np.linalg.norm(self.lv_path * (i + 0.5) / self.npts) for i in range(self.npts)])
        
        if 'nprocs' in args:
            self.nprocs = int(args['nprocs'])
        else:
            self.nprocs = 1
        
        self.tip_disp = None
        self.tip_scaler = None
        ldos_data = None

        if 'tip_disp' in args:
            self.tip_disp = float(args['tip_disp'])
        if 'tip_scaler' in args and 'ldos_data' in args and 'tip_dist' in args:
            self.tip_scaler = float(args['tip_scaler'])
            ldos_data = args['ldos_data']
            self.tip_dist = float(args['tip_dist'])
        # Extracting tip heights
        tip_heights = np.zeros(self.npts)

        for i in range(self.npts):
            if self.tip_disp is not None:
                pos = np.array([0.0, 0.0, np.max(self.coord[:, 2]) + self.tip_disp])
            elif self.tip_scaler is not None and ldos_data is not None and self.tip_dist is not None:
                tip_height = (ldos_data[i][1] * self.tip_scaler) + self.tip_dist
                pos = np.array([0.0, 0.0, np.max(self.coord[:, 2]) + tip_height])
                tip_heights[i] = tip_height
            else:
                raise ValueError("Either 'tip_disp' or 'tip_scaler' with 'ldos_data' must be provided.")

            if 'path_atoms' not in args:
                pos += self.lv_origin + self.lv_path * (i / (self.npts - 1))
            else:
                pos[:2] += self.path_coord[i]

            self.x[i], self.y[i], self.z[i] = pos[0], pos[1], pos[2]

        # Plotting the tip heights vs distance along LDOS line
        plt.figure(figsize=(10, 6))
        plt.plot(self.path_distance, tip_heights, marker='o', linestyle='-', color='b')
        plt.title('Tip Heights vs. Distance Along LDOS Line')
        plt.xlabel('Distance Along LDOS Line')
        plt.ylabel('Tip Height')
        plt.grid(True)
        plt.show()
            
        # The list exclude includes the indices of atoms to exclude from LDOS integration
        self.exclude = []
        if 'exclude' in args:
            self.exclude_args = args['exclude']
            counter = 0
            for i in self.atomtypes:
                if i in args['exclude']:
                    for j in range(self.atomnums[self.atomtypes.index(i)]):
                        self.exclude.append(counter)
                        counter += 1
                else:
                    counter += self.atomnums[self.atomtypes.index(i)]
        print(str(len(self.exclude)) + ' atoms excluded from LDOS averaging')
        
        if 'unit_cell_num' in args:
            self.unit_cell_num = args['unit_cell_num']
        
        self.periodic_coord = []
        for i in range(-1 * self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-1 * self.unit_cell_num, self.unit_cell_num + 1):
                for k in self.coord:
                    self.periodic_coord.append(k + self.lv[0] * i + self.lv[1] * j)
        self.periodic_coord = np.array(self.periodic_coord)
        
        for i in range(len(self.energies)):
            if self.energies[i] < emin:
                self.estart = i
            if self.energies[i] > emax:
                self.eend = i
                break
        else:
            print('specified emax exceeds maximum energy in DOSCAR.')
            print('integrating from {} to {} V'.format(self.emin, self.energies[-1]))
        
        self.ldos = np.zeros((len(self.orbitals), self.npts, self.eend - self.estart))
        
        if 'phi' in args and args['phi'] != 0:
            self.phi = float(args['phi'])
            self.K = np.array([tunneling_factor(abs(i), abs(i), self.phi) for i in self.energies[self.estart:self.eend]])
        else:
            self.K = np.array([1.0 for _ in range(self.estart, self.eend)])
            self.phi = 0.0
        
        for i in range(self.npts):
            if self.tip_disp is not None:
                pos = np.array([0.0, 0.0, np.max(self.coord[:, 2]) + self.tip_disp])
            elif self.tip_scaler is not None and ldos_data is not None and self.tip_dist is not None:
                tip_height = (ldos_data[i][1] * self.tip_scaler)+self.tip_dist
                pos = np.array([0.0, 0.0, np.max(self.coord[:, 2]) + tip_height])
            else:
                raise ValueError("Either 'tip_disp' or 'tip_scaler' with 'ldos_data' must be provided.")
            
            if 'path_atoms' not in args:
                pos += self.lv_origin + self.lv_path * (i / (self.npts - 1))
            else:
                pos[:2] += self.path_coord[i]
            
            self.x[i], self.y[i], self.z[i] = pos[0], pos[1], pos[2]
        
        start = time()
        # Executes ldos integration in parallel on a ProcessPool of self.nprocs processors
        if self.nprocs > 1:
            #from pathos.multiprocessing import ProcessPool
            pool = ProcessPool(self.nprocs)
            output = pool.map(self.integrator, [i for i in range(self.npts)])
            self.ldos = np.sum(output, axis=0)
            pool.close()
        # Executes ldos integration on a single processor
        else:
            for i in range(self.npts):
                pos = np.array([self.x[i], self.y[i], self.z[i]])
                counter = 1
                for k in self.periodic_coord:
                    if counter == sum(self.atomnums) + 1:
                        counter = 1
                    if counter - 1 not in self.exclude:
                        posdiff = np.linalg.norm(pos - k)
                        sf = np.exp(-1.0 * posdiff * self.K * 1.0e-10)
                        for l in range(len(self.dos[counter])):
                            self.ldos[l][i] += self.dos[counter][l][self.estart:self.eend] * sf
                    counter += 1
        print('total time to integrate {} points: {} seconds on {} processors'.format(self.npts * (self.eend - self.estart), time() - start, self.nprocs))

    # Other methods...
# Example usage:
# Assuming `ldos_data` is obtained from `plot_energy_slice` method
# test = ldos_line('C:/Users/Benjamin/Documents/VASP/NHC/iPr/SAM/adatoms/rect_staggered/mid/spinorb/freegold1/freeegold2/freegold3/kpoints551')
# test.parse_VASP_output()
# ldos_data = test.plot_energy_slice(0.2)
# test.calculate_ldos(64, 0.21, 0.19, np.array([-0.34206, -1.12685, 0]), np.array([0.5, 0.875, 0]), tip_scaler=15, ldos_data=ldos_data)

    #performs integration at single point of the x,y grid when run in parallel
    def integrator(self,i):
        from numpy import array
        pos=array([self.x[i],self.y[i],self.z[i]])
        temp_ldos=zeros((len(self.orbitals),self.npts,self.eend-self.estart))
        counter=1
        for k in self.periodic_coord:
            if counter==sum(self.atomnums)+1:
                    counter=1
            if counter-1 not in self.exclude:
                posdiff=norm(pos-k)
                sf=exp(-1.0*posdiff*self.K*1.0e-10)
                for l in range(len(self.dos[counter])):
                    temp_ldos[l][i]+=self.dos[counter][l][self.estart:self.eend]*sf
            counter+=1
        
        return temp_ldos
    
    def normalize_position_slices(self, norm_range='full'):
        self.zero_e = np.argmin(abs(self.energies)) - self.estart
        
        # Print shapes to ensure they are 3D
        #print(f"self.ldos shape: {self.ldos.shape}")
        #print(f"self.copy_ldos shape: {self.copy_ldos.shape}")

        self.copy_ldos = copy.deepcopy(self.ldos)  # Use deepcopy to ensure a true copy

        if norm_range == 'full':
            for i in range(self.npts):
                for j in range(np.shape(self.ldos)[0]):
                    norm_value = np.sum(self.copy_ldos[:, i, :self.eend-self.estart])
                    #print(f"Normalization value (full) at i={i}, j={j}: {norm_value}")
                    if norm_value != 0:  # Check for non-zero sum
                        self.ldos[j, i, :] /= norm_value
                   # else:
                        #print(f"Skipping normalization at i={i}, j={j} due to zero normalization value.")
        elif norm_range == 'positive':
            for i in range(self.npts):
                for j in range(np.shape(self.ldos)[0]):
                    norm_value = np.linalg.norm(self.copy_ldos[:, i, self.zero_e:self.eend-self.estart])
                    #print(f"Normalization value (positive) at i={i}, j={j}: {norm_value}")
                    if norm_value != 0:
                        self.ldos[j, i, :] /= norm_value
                 #   else:
                        #print(f"Skipping normalization at i={i}, j={j} due to zero normalization value.")
        elif norm_range == 'negative':
            for i in range(self.npts):
                for j in range(np.shape(self.ldos)[0]):
                    norm_value = np.sum(self.copy_ldos[:, i, :self.zero_e])
                    #print(f"Normalization value (negative) at i={i}, j={j}: {norm_value}")
                    if norm_value != 0:
                        self.ldos[j, i, :] /= norm_value
                  #  else:
                        #print(f"Skipping normalization at i={i}, j={j} due to zero normalization value.")

                  
    def smear_spatial(self,dx):
        dx/=self.path_distance[1]
        for i in range(self.eend-self.estart):
            self.ldos[:,i]=gaussian_filter(self.ldos[:,i],dx,mode='constant')
        
    #plots the ldos map and overlaid atoms on size+1 periodic cells
    def plot_map(self,norm_range=False,dx=0,**args):
        if 'cmap' in args:
            self.cmap=args['cmap']
            
        if 'orbitals' in args:
            orbitals_to_plot=args['orbitals']
        else:
            orbitals_to_plot=[i for i in range(len(self.orbitals))]
        if len(orbitals_to_plot)==len(self.orbitals):
            self.orbitals=['all']
        else:
            self.orbitals=[self.orbitals[i] for i in orbitals_to_plot]
        self.ldos=sum([self.ldos[i] for i in orbitals_to_plot])
        
        if 'normalize_ldos' in args:
            normalize_ldos=args['normalize_ldos']
        else:
            normalize_ldos=True
            
        if norm_range:
            self.normalize_position_slices(norm_range=norm_range)
        if dx!=0:
            self.smear_spatial(dx)
            
        self.ldosfig,self.ldosax=plt.subplots(1,1)
        
        #plots the ldos
        if normalize_ldos:
            ldosmap=self.ldosax.pcolormesh(array([self.energies[self.estart:self.eend] for i in range(self.npts)]),array([[self.path_distance[i] for j in range(self.eend-self.estart)] for i in range(self.npts)]),self.ldos/np.max(self.ldos),cmap=self.cmap,shading='nearest')
        else:
            ldosmap=self.ldosax.pcolormesh(array([self.energies[self.estart:self.eend] for i in range(self.npts)]),array([[self.path_distance[i] for j in range(self.eend-self.estart)] for i in range(self.npts)]),self.ldos,cmap=self.cmap,shading='nearest')
                
        if 'show_colorbar' in args:
            self.ldosfig.colorbar(ldosmap)
        
        self.ldosax.set(ylabel=r'distance along ldos line / $\AA$')
        self.ldosax.set(xlabel='energy - $E_f$ / eV')
        self.ldosax.set(title=r'{} $\AA$ | {} $\AA$ | {} $\AA$ | \ncontributing orbitals: {}'.format(self.tip_disp, self.tip_scaler, self.tip_dist, ', '.join(self.orbitals)))
        self.ldosfig.subplots_adjust(top=0.9)
        self.ldosfig.show()
        
    #take a slice of the ldos plot at a specific point in the path
    #options for specifying are: x (positional, direct), y (positional, direct), or pos (position along self.path_distance)

        
    #take a slice of the ldos at a given energy to see spatial dependance
    def plot_energy_slice(self, ref_energy):
        if self.energy_slices == 0:
            self.eslice_fig, self.eslice_ax = plt.subplots(1, 1)
            self.eslice_fig.show()
            
        mindiff = self.energies[-1] - self.energies[0]
        for i in range(self.estart, self.eend):
            if abs(self.energies[i] - ref_energy) < mindiff:
                mindiff = abs(self.energies[i] - ref_energy)
                ref_index = i
        
        # Extract the LDOS slice at the reference energy
        ldos_slice = self.ldos[:, ref_index - self.estart]

        # Normalize the LDOS slice to a maximum of 1
        max_value = np.max(ldos_slice)
        if max_value != 0:
            ldos_slice /= max_value

        # Plot the normalized LDOS slice along the path
        tempvar = self.eslice_ax.plot(self.path_distance, ldos_slice, label=f'Energy Slice at {ref_energy:.2f} eV')
        self.ldosax.plot(np.array([self.energies[ref_index] for i in range(2)]), 
                         np.array([self.path_distance[0], self.path_distance[-1]]), 
                         c=tempvar[0].get_color(), linewidth=6)
        self.eslice_ax.set(ylabel='normalized tunneling probability')
        self.eslice_ax.set(xlabel=r'distance along ldos line / $\AA$')
        self.ldosfig.canvas.draw()
        self.eslice_fig.canvas.draw()
        self.energy_slices += 1

        # Generate the list of (distance along ldos line, normalized tunneling probability)
        ldos_data = list(zip(self.path_distance, ldos_slice))

        return ldos_data
    
    
    def plot_summed_ldos(self):
        if self.energy_sums == 0:
            self.esum_fig, self.esum_ax = plt.subplots(1, 1)
            self.esum_fig.show()
    
        # Initialize an array to hold the integrated LDOS values
        ldos_integral = np.zeros(self.npts)
    
        # Loop over each spatial position
        for i in range(self.npts):
            # Take a position slice
            ldos_slice = self.ldos[i, :]
    
            # Integrate the LDOS slice using simpsonon's rule
            ldos_integral[i] = simpson(ldos_slice, self.energies[self.estart:self.eend])
    
        # Normalize the integrated LDOS to a maximum of 1
        max_value = np.max(ldos_integral)
        if max_value != 0:
            ldos_integral /= max_value
    
        # Plot the normalized integrated LDOS along the path
        self.esum_ax.plot(self.path_distance, ldos_integral, label='Summed and Integrated LDOS')
        self.esum_ax.set(ylabel='normalized tunneling probability')
        self.esum_ax.set(xlabel=r'distance along LDOS line / $\AA$')
        self.esum_ax.legend()
        self.esum_fig.canvas.draw()
    
        # Generate the list of (distance along LDOS line, normalized tunneling probability)
        ldos_data = list(zip(self.path_distance, ldos_integral))
    
        return ldos_data

    # Other methods...

# Example usage:


if __name__=='__main__':
    sys.path.append(getcwd())
    exclude=['none']
    nprocs=1
    #a 15 Angstrom tip displacement gives realistic images at low voltage
    tip_disp=15.0
    #sets the number of unit cells considered along each lattice vector
    #4 gives good results when the sampling displacement is on the same order of magnitude as the lattice vector magnitudes
    unit_cell_num=4
    npts=1
    phi=0
    try:
        opts,args=getopt.getopt(sys.argv[1:],'e:n:x:p:t:u:w:',['erange=','npts=','exclude=','processors=','tip_disp=','num_unit_cells=','work_function='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-e','--erange']:
            emin=min([float(k) for k in j.split(',')])
            emax=max([float(k) for k in j.split(',')])
        if i in ['-n','--npts']:
            npts=int(j)
        if i in ['-x','--exclude']:
            exclude=[str(k) for k in j.split(',')]
        if i in ['-p', '--processors']:
            nprocs=int(j)
        if i in ['-t','--tip_disp']:
            tip_disp=float(j)
        if i in ['-u','--unit_cell_num']:
            unit_cell_num=int(j)
        if i in ['-w','-work_function']:
            phi=float(j)
    if exists('./DOSCAR'):
        main=ldos_line('./')
        main.parse_VASP_output()
        main.calculate_ldos(npts,emax,emin,exclude=exclude,nprocs=nprocs,tip_disp=tip_disp,unit_cell_num=unit_cell_num,phi=phi)
        main.write_ldos()

#helper function to generate path from VESTA text
#simply click the atoms you would like to slice through in the order you would like to slice them
#any duplicates will be discarded
#then copy all the text from the VESTA window and paste as a single string into the argument of this function
#the path list that can be supplied to the slice_path class is returned
def create_path_from_VESTA(text):
    path=[]
    text=text.split()
    counter=-1
    for i in text:
        if counter<0 and i=='Atom:':
            counter=0
            tempvar=[]
        elif counter==0:
            for j in path:
                if int(i)==j[0]:
                    counter=-1
                    break
            else:
                tempvar.append(int(i))
                counter+=1
        elif counter in [1,2]:
            counter+=1
        elif counter in [3,4]:
            tempvar.append(int(np.floor(float(i))))
            counter+=1
            if counter==5:
                path.append(tempvar)
                counter=-1

    return path