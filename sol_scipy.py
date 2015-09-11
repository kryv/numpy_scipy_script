#
# sol_scipy.py : particle tracking in a solenoid
#                   code by Kei F.
from numpy import *
from scipy.integrate import *
from scipy.optimize import *
from scipy.interpolate import *

sR   = 0.08
sl   = 0.3
Lz   = 20.0
cl   = 2.99792458e8
ms   = 238.0*1.66053892e-27
qc   = 1.60217657e-19
uc   = 34.0
en   = 3.0e6
gmm  = qc*en/(ms*cl**2.0)+1 
vz0   = cl*sqrt(1-1/gmm**2.0)
#bz0  = 1.8184449309034985 #edge
bz0  = 1.7647058823529411 #center
bfac = 0.0
twpi = 2.0*pi

def brf(th,r,z):
    global twpi,sR,sL,bz0,bfac
    return(bfac/twpi*(
        (sR*cos(th))/sqrt((z-sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))
       -(sR*cos(th))/sqrt((z+sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))
               )/bz0)
               
def br(r,z):  
    return(quad(brf,0.0,2.0*pi,args=(r,z))[0])


def bzf(th,r,z):
    global twpi,sR,sL,bz0
    return(bfac/twpi*
            (sR**2.0 - sR*r*cos(th))/(sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))*
            ((z+sl/2.0)/sqrt((z+sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th)) -
             (z-sl/2.0)/sqrt((z-sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))
            )/bz0)
            
def bz(r,z):  
    return(quad(bzf,0.0,2.0*pi,args=(r,z))[0])
    

def atf(th,r,z):
    global twpi,sR,sL,bz0
    return(bfac*(r*sR**2.0)/twpi*
            (sin(th)**2.0)/(sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))*
            ((z+sl/2.0)/sqrt((z+sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th)) -
             (z-sl/2.0)/sqrt((z-sl/2.0)**2.0 + sR**2.0 + r**2.0 - 2.0*sR*r*cos(th))
            )/bz0)
            
def at(r,z):  
    return(quad(atf,0.0,2.0*pi,args=(r,z))[0])
    

def kp(yvec):
    global ms,cl
    xx,xp,yy,yp,zz,zp = yvec    
    rr = sqrt(xx**2.0 + yy**2.0)
    beta = sqrt(xp**2.0+yp**2.0+zp**2.0)/cl
    gamma = 1.0/sqrt(1.0-beta**2.0)
    
    pkin = ms*gamma*(xx*yp-yy*xp)
    ppot = uc*qc*rr*at(rr,zz)
    pthn = pkin + ppot
    prr3 = ((pthn/ms/cl/gamma/beta)**2.0)/rr**3.0
    return(pkin,ppot,pthn,prr3)
    
nt = 3001 #total step
tvec = linspace(0,Lz/vz0,nt)
#initial values  [x, vx, y, vy, z, vz0]    
init = [0.02,100.0,0.02,0.0,-Lz/2.0,vz0]

def deriv(t,yvec):
    global qc,gmm,ms
    fr = [0]*6
    xx,xp,yy,yp,zz,zp = yvec
    rr = sqrt(xx**2.0 + yy**2.0)
    fr[0] = xp
    fr[1] = (uc*qc/ms/gmm)*( bz(rr,zz)*yp-br(rr,zz)*yy*zp/rr)
    fr[2] = yp
    fr[3] = (uc*qc/ms/gmm)*(-bz(rr,zz)*xp+br(rr,zz)*xx*zp/rr)
    fr[4] = zp
    fr[5] = (uc*qc/ms/gmm)*br(rr,zz)*(yy*xp - xx*yp)/rr
    return fr

ans=ode(deriv)
ans.set_integrator('dopri5') #forth order runge-kutta
#ans.set_integrator('dop853') #eighth order runge-kutta
ans.set_initial_value(init)

hst=empty((nt,6))
hst[0] = init
k=1
while ans.successful() and ans.t <  Lz/vz0 :
    ans.integrate(tvec[k])
    hst[k]=ans.y
    k += 1

xx = hst[:,0] 
vx = hst[:,1]
yy = hst[:,2]
vy = hst[:,3]
zz = hst[:,4]
vz = hst[:,5]
   
pt = array( [ kp(hst[i]) for i in range(nt)])

rp = (xx*vx+yy*vy)/sqrt(xx*xx+yy*yy)/vz
tot = sum(pt[:,3]*(Lz/vz0/(nt-1))*vz)

drp = rp[-1]-rp[0]-tot
print "Delta r' |focus = " + str(drp)

#output data list 0:xx   1:vx   2:yy   3:vy   4:zz   5:vz   
#                 6:mass*gamma*(xx*vy - yy*vx)
#                 7:q*r*A_theta 
#                 8:p_theta = mass*gamma*(xx*vy - yy*vx) + q*r*A_theta
savetxt("alldata.dat",transpose((xx,vx,yy,vy,zz,vz,pt[:,0],pt[:,1],pt[:,2])))