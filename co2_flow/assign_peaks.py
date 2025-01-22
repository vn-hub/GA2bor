import numpy as np

class Peak2f:
    def __init__(self, max_val_x, min_left_x, min_right_x, s2_max, s2_ampl):
        self.max_x = max_val_x
        self.min_lx = min_left_x
        self.min_rx = min_right_x

        self.s2_max = s2_max
        self.s2_ampl = s2_ampl


def VerifyPeaksLength(max_xs, min_xs, peak_objects):
    if 2*len(max_xs) == len(min_xs) or len(max_xs) == 2*len(min_xs):
        print('Peaks matched/consistent definition!')
        print('Total number of peaks identified in 2f-WMS signal is: ' + str(len(peak_objects)))
    else:
        print('Peak mismatch/inconsistent definition!')
        print('Positions of detected local maxima in 2f-WMS signal are: ' + str(max_xs))
        print('Positions of detected local minima in 2f-WMS signal are: ' + str(min_xs))
        exit(1)

#na vstupu tri body, overujeme ze prostredni je uprostred
def IsValidPeak(x1,x2,x3):
    if x1 < x2 and x2 < x3:
        return True
    else:
        return False

#vytvori pole peaku a overi ze jsou platne
def AssignPeaks(set1_x, set2_x, set1_y, set2_y):
    peaks = []
    if len(set1_x) < len(set2_x):
        for i in range(len(set1_x)):
            x_left = set2_x[i*2]
            x_right = set2_x[i*2 + 1]

            if IsValidPeak(x_left, set1_x[i], x_right):
                ampl = np.abs((set2_y[i*2] + set2_y[i*2+1])/2 - set1_y[i])
                p = Peak2f(set1_x[i],x_left,x_right,np.abs(set1_y[i]),ampl)
                peaks.append(p)

    elif len(set2_x) < len(set1_x):
        for i in range(len(set2_x)):
            x_left = set1_x[i*2]
            x_right = set1_x[i*2 + 1]

            if IsValidPeak(x_left, set2_x[i], x_right):
                ampl = np.abs((set1_y[i*2] + set1_y[i*2+1])/2 - set2_y[i])
                p = Peak2f(set2_x[i],x_left,x_right,abs(set2_y[i]),ampl)
                peaks.append(p)

    return peaks

#funkce ulozi experimentalni spektrum v rozsahu n*(vzdalenost minim)
def SavePeakSpectra(peaks, x, y, in_file, n=2):
    for peak_id in range(len(peaks)):
        peak = peaks[peak_id]
        diff = n*np.abs(peak.min_lx - peak.min_rx)

        delta = diff
        #overit, ze vzdalenost od stredu peaku do zacatku dat je max delta
        if peak.min_lx - delta < x[0]:
            delta = np.abs(peak.min_lx - x[0])

        #overit konec dat
        if peak.min_rx + delta > x[-1]:
            delta = np.abs(x[-1] - peak.min_rx)

        #overit predchozi peak
        if peak_id > 0:
            peak_prec = peaks[peak_id - 1]
            d_px = np.abs(peak.min_lx - peak_prec.min_rx)/2
            if d_px < delta:
                delta = d_px

        #overit nasledny peak
        if len(peaks) > peak_id + 1:
            peak_succ = peaks[peak_id + 1]
            d_px = np.abs(peak.min_rx - peak_succ.min_lx)/2
            if d_px < delta:
                delta = d_px

        #najdi pocatecni index
        pos_start = 0
        x_left = peak.min_lx - delta
        x_right = peak.min_rx + delta
        
        while(x[pos_start] < x_left):
            pos_start += 1

        #najdi koncovy index
        pos_end = pos_start
        while(x[pos_end] < x_right):
            pos_end += 1

        print("Peak" + str(peak_id+1) + " centered at: " + str(peak.max_x) + "(left minimum: " + str(peak.min_lx) + "; right minimum: " + str(peak.min_rx)+ ")")

        f = open(in_file.replace(".txt","_Peak" + str(peak_id+1) + ".txt"),"w")
        #header
        f.write("##Delta: " + str(delta) + "\tInterval: " + str(x[pos_start]) + "-" + str(x[pos_end]) + "\n")
        #data
        for j in range(pos_start, pos_end+1):
            f.write(str(x[j]) + "," + str(y[j]) + "\n")
 
        f.close()