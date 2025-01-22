#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

from pybaselines import Baseline, utils

import lmfit as lm
from lmfit import Model, Parameters, Parameter, report_fit, minimize, Minimizer

from scipy.signal import find_peaks

from assign_peaks import *


InputFile = 'C1(2f)_2023_05_31_17-55.txt'
Dataset = np.loadtxt(InputFile, skiprows= 5, delimiter=',')
raw_x = Dataset[:, 0]
raw_y = Dataset[:, 1]
#plt.plot(raw_x, raw_y)

#Number_of_Samples_per_Second
SpS = int(len(raw_x)/np.max(raw_x))
print("!Sampling rate of the datases is " + str(SpS) + " Samples/s!" )

Dataset = np.loadtxt(InputFile, skiprows= 5, delimiter=',')
x = Dataset[4000:24000, 0]
y = Dataset[4000:24000, 1]
#plt.plot(y)

Out1 = PdfPages('out2f_exp.pdf')
Out2 = PdfPages('out2f_fit.pdf')

baseline_fitter = Baseline(x_data=x)

half_window_1 = 333
half_window_2 = 500
half_window_3 = 1000
half_window_4 = 2000

fit_1, params_1 = baseline_fitter.std_distribution(y, half_window_1, smooth_half_window=None)
fit_2, params_2 = baseline_fitter.std_distribution(y, half_window_2, smooth_half_window=None)
fit_3, params_3 = baseline_fitter.std_distribution(y, half_window_3, smooth_half_window=None)
fit_4, params_4 = baseline_fitter.std_distribution(y, half_window_4, smooth_half_window=None)

mask_1 = params_1['mask']
mask_2 = params_2['mask']
mask_3 = params_3['mask']
mask_4 = params_4['mask']

_, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, gridspec_kw={'hspace': 0})
ax1.plot(x[mask_1], y[mask_1], "x")[0]
ax1.plot(x, y)
ax1.plot(x, fit_1, label=f'half_window={half_window_1}')
ax1.legend()
ax2.plot(x[mask_2], y[mask_2], "x")[0]
ax2.plot(x, y)
ax2.plot(x, fit_2, label=f'half_window={half_window_2}')
ax2.legend()
ax3.plot(x[mask_3], y[mask_3], "x")[0]
ax3.plot(x, y)
ax3.plot(x, fit_3,  label=f'half_window={half_window_3}')
ax3.legend()
ax4.plot(x[mask_4], y[mask_4], "x")[0]
ax4.plot(x, y)
ax4.plot(x, fit_4, label=f'half_window={half_window_4}')
ax4.legend()
plt.show()

###Background subtraction
#y=y-fit_3
y=y-fit_4

valleys, _ = find_peaks(-y, height=0, distance=500, width=250)
np.diff(valleys)
#plt.plot(-y)
#plt.plot(valleys, -y[valleys], "x")
#plt.show()

xval = x[valleys]
yval = y[valleys]

peaks, _ = find_peaks(y, height=0.001, distance=2500, width=500)
np.diff(peaks)
#plt.plot(y)
#plt.plot(peaks, y[peaks], "x")
#plt.show()

xpik = x[peaks]
ypik = y[peaks]

plt.figure(1)

line1, = plt.plot(x, y+fit_4, 'black', linewidth=3, label='Experiment')
line2, = plt.plot(x, fit_4, 'orange', linewidth=3, label='Background')
line3, = plt.plot(x, y, 'blue', linewidth=3, label='Experiment - Background')
plt.axhline(linewidth=2, linestyle='dotted', color='grey')

point1, = plt.plot(xpik, ypik, 'ro', linewidth=3, label='Peak position (local maxima)')
point2, = plt.plot(xval, yval, 'go', linewidth=3, label='Peak position (local minima)')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=1)})
plt.xlabel(r"Scan time [s]")
plt.ylabel(r"2f-WMS signal [a.u.]")

plt.tight_layout()
a4 = plt.gcf()
a4.set_size_inches([10,7])
a4.savefig(Out1, format='pdf')
plt.show()

Out1.close()

#vytvoreni pole objektu peaku
peak_objects = AssignPeaks(xpik, xval, ypik, yval)
numpeak = len(peak_objects)
#overeni delek poli minim a maxim
VerifyPeaksLength(xpik, xval, peak_objects)
#vypsat
SavePeakSpectra(peak_objects, x, y, InputFile, 2)


def lorentzianA(x, a0, a1, a2):
    return a0/(np.pi*a2*(1+((x-a1)/a2)**2))

def D2lorentzianA(x, a0, a1, a2):
    return -2*a0*a2*(a2**2-3*(a1-x)**2)/(np.pi*(a2**2+(a1-x)**2)**3)

def gaussianA(x, a0, a1, a2):
    sigma = a2/np.sqrt(2*np.log(2))
    return a0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*sigma**2))

def D2gaussianA(x, a0, a1, a2):
    sigma = np.sqrt(2*np.log(2))*a2
    poly2 = ((x-a1)**2-sigma**2)/sigma**4
    return poly2*a0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*sigma**2))


result=[None]*numpeak
par=[None]*numpeak
resdata=[None]*numpeak
ofst=[None]*numpeak
minner=[None]*numpeak
res2ult=[None]*numpeak
fin2ual=[None]*numpeak
res2data=[None]*numpeak
par2am=[None]*numpeak
gabor2=[None]*numpeak
gabor4=[None]*numpeak
gabor6=[None]*numpeak
gabor8=[None]*numpeak

for i in range(1,numpeak+1):
    j=i-1
    
    InputFilePeaks = InputFile.replace(".txt", '_Peak') + str(i) + '.txt'
    Dataset = np.loadtxt(InputFilePeaks, skiprows= 0 + 1 + 2 + 3 + 4 , delimiter=',')
    x = Dataset[:, 0]
    y = Dataset[:, 1]

    Out3i = PdfPages('out_peakfit' + str(i) + '.pdf')
    
    params = lm.Parameters()
    params.add('a0', value=np.maximum(np.abs(np.amax(-y)),np.abs(np.amin(-y))), min=10*np.amin(-y), max=10*np.amax(-y))
    params.add('a1', value=np.amin(x)-(np.amax(x)-np.amin(x))/2, min=np.amin(x), max=np.amax(x))
    params.add('a2', value=(np.amax(x)-np.amin(x))/5, min=(np.amax(x)-np.amin(x))/10, max=(np.amax(x)-np.amin(x))/2)

    #model = lm.Model(D2gaussianA)
    model = lm.Model(D2lorentzianA)
     #result[j] = model.fit(y, params, x=x, method='leastsq')
    result[j] = model.fit(y, params, x=x, method='differential_evolution')
    
    lm.report_fit(result[j])

    par[j] = result[j].best_values
    resdata[j] = result[j].residual
    ofst[j] = np.mean(resdata[j])
    
    def residual(params, x, resdata):
        "Gabor polynomial"
        params['a0']= Parameter(name = 'a0', value = par[j]['a0'] , vary = False)
        params['a1']= Parameter(name = 'a1', value = par[j]['a1'] , vary = False)
        params['a2']= Parameter(name = 'a2', value = par[j]['a2'] , vary = False)

        off = params['off']
        params['off']= Parameter(name = 'off', value = ofst[j] , vary = False)
        k2 = params['k2']
        k4 = params['k4']
        k6 = params['k6']

        #k8 = params['k8']
        #fi8 = params['fi8']

        fi6 = params['fi6']
        fi4 = params['fi4']
        fi2 = params['fi2']
        
        a0 = params['a0']
        a1 = params['a1']
        a2 = params['a2']

        w = params['w']

        sigma = a2/np.sqrt(2*np.log(2))
        #gabor8 = np.cos(2*np.pi*(0/2)*(x-a1)/(2*a2)+fi8)*a0/((w*sigma)*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*(w*sigma)**2))
        gabor6 = np.sin(2*np.pi*(3/2)*(x-a1)/(2*a2)+fi6)*a0/((w*sigma)*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*(w*sigma)**2))
        gabor4 = np.cos(2*np.pi*(2/2)*(x-a1)/(2*a2)+fi4)*a0/((w*sigma)*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*(w*sigma)**2))
        gabor2 = np.sin(2*np.pi*(1/2)*(x-a1)/(2*a2)+fi2)*a0/((w*sigma)*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*(w*sigma)**2))
        
        #return off + k8*gabor8 + k6*gabor6 + k4*gabor4 + k2*gabor2 - resdata
        return off + k6*gabor6 + k4*gabor4 + k2*gabor2 - resdata
    
        # create a set of Parameters
    
    params.add('off', min=0, max=par[j]['a0'])
    params.add('k6', value=1.0, min=0.1, max=20.0, vary = True)
    params.add('k4', value=1.0, min=0.1, max=20.0, vary = True)
    params.add('k2', value=1.0, min=0.1, max=20.0, vary = True)
    #params.add('k8', value=1.0, min=-2.5, max=+2.5, vary = True)
    #params.add('fi8', value=0.0, min=-np.pi, max=np.pi, vary = False)
    params.add('fi6', value=0.0, min=-np.pi, max=np.pi, vary = True)
    params.add('fi4', value=0.0, min=-np.pi, max=np.pi, vary = True)
    params.add('fi2', value=0.0, min=-np.pi, max=np.pi, vary = True)
    params.add('w', value=1, min=1/3, max=4/3, vary = True)

       
    # do fit, here with leastsq model
    minner[j] = Minimizer(residual, params, fcn_args=(x, resdata[j]))
    res2ult[j] = minner[j].minimize(method = 'leastsq')
    # res2ult[j] = minner[j].minimize(method = 'differential_evolution')

    par2am[j] = res2ult[j].params
    #par2am[j].pretty_print()

    offj=par2am[j]['off'].value
    a0j=par2am[j]['a0'].value
    a1j=par2am[j]['a1'].value
    a2j=par2am[j]['a2'].value
    k6j=par2am[j]['k6'].value
    k4j=par2am[j]['k4'].value
    k2j=par2am[j]['k2'].value
    #k8j=par2am[j]['k8'].value
    #fi8j=par2am[j]['fi8'].value
    fi6j=par2am[j]['fi6'].value
    fi4j=par2am[j]['fi4'].value
    fi2j=par2am[j]['fi2'].value
    sigmaj = a2j/np.sqrt(2*np.log(2))
    wj=par2am[j]['w'].value

    #gabor8[j] = np.cos(0*np.pi*(x-a1j)/(2*a2j)+fi8j)*k8j*a0j/(wj*sigmaj*np.sqrt(2*np.pi))*np.exp(-(x-a1j)**2/(2*(wj*sigmaj)**2))
    gabor6[j] = np.sin(3*np.pi*(x-a1j)/(2*a2j)+fi6j)*k6j*a0j/(wj*sigmaj*np.sqrt(2*np.pi))*np.exp(-(x-a1j)**2/(2*(wj*sigmaj)**2))
    gabor4[j] = np.cos(2*np.pi*(x-a1j)/(2*a2j)+fi4j)*k4j*a0j/(wj*sigmaj*np.sqrt(2*np.pi))*np.exp(-(x-a1j)**2/(2*(wj*sigmaj)**2))
    gabor2[j] = np.sin(1*np.pi*(x-a1j)/(2*a2j)+fi2j)*k2j*a0j/(wj*sigmaj*np.sqrt(2*np.pi))*np.exp(-(x-a1j)**2/(2*(wj*sigmaj)**2))

    res2data[j] = res2ult[j].residual
    
    # calculate final result
    fin2ual[j] = resdata[j] + res2data[j]
    
    # write error report
    report_fit(res2ult[j])
    
    fig2 = plt.figure(2)
    
    ax1=plt.subplot(311)
    line2, = plt.plot(x, y, 'grey', linewidth=2, label= r'Experiment ($S_{2f}^{exp}$)')
    line3, = plt.plot(x, result[j].best_fit, 'blue', linewidth=2, label= r'$0^{th}$ ord. approx. ($\frac{d^{2}L}{d\tau}$)')
    line3f, = plt.plot(x, result[j].best_fit-fin2ual[j], 'red', linewidth=2, label= r'$1^{st}$ ord. approx. ($S_{2f}^{calc}$)')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1)})
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=3)
    plt.xlabel(r"Scan time [s]")
    plt.ylabel(r"2f-WMS signal [a.u.]")
    plt.axis([1.0*np.amin(x), 1.0*np.amax(x), 1.2*np.amin(y), 1.2*np.amax(y),])
    plt.grid(True)

    ax2=plt.subplot(312, sharex=ax1)
    line4, = plt.plot(x, -result[j].residual, 'blue',linewidth=2, label=r'Residual of the $0^{th}$ ord. approx. ($R_{nf}$)')
    line5, = plt.plot(x, -fin2ual[j], 'magenta', linewidth=2, label=r'Sum of $1^{st}$ ord. correction terms ($\Gamma_0 + \Gamma_1 + M$)')
    plt.legend(handler_map={line4: HandlerLine2D(numpoints=1)})
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=1)
    plt.xlabel(r"Scan time [s]")
    plt.ylabel(r"Residual [a.u.]")
    plt.grid(True)
    
    ax3=plt.subplot(313, sharex=ax1)
    line6, = plt.plot(x, res2ult[j].residual, 'red',linewidth=2, label=r'Residual of the $1^{st}$ ord. approx. ($\rho_{nf}$)')
    line7, = plt.plot(x, -gabor2[j], 'pink',linewidth=2, label= r'Gabor function ($\Gamma_{0,sin}$)')
    line8, = plt.plot(x, -gabor4[j], 'yellow',linewidth=2, label= r'Gabor function ($\Gamma_{1,cos}$)')
    line9, = plt.plot(x, -gabor6[j], 'orange',linewidth=2, label=r'Gabor function ($\Gamma_{1,sin}$)')
    #line10, = plt.plot(x, gabor8[j], 'cyan',linewidth=2, label='Gabor function (n=8)')
    plt.legend(handler_map={line6: HandlerLine2D(numpoints=1)})
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=2)
    plt.xlabel(r"Scan time [s]")
    plt.ylabel(r"Residual [a.u.]")
    plt.grid(True)
    
    plt.tight_layout()
    a4 = plt.gcf()
    a4.set_size_inches([7,10])
    a4.savefig(Out2, format='pdf')
    a4.savefig(Out3i, format='pdf')
    plt.show()

    Out3i.close()
    
Out2.close()
