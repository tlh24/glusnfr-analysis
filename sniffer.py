import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import math
import pdb
import torch
from itertools import repeat

Abhi = True
nShuf = 50000 # default is 10000

if Abhi: 
    fil = pd.read_excel('yGluSnFR Summary_updated_June18th.xlsx',sheet_name='Sheet1')

    wildTypeSeq = 'GSAAGSTLDKIAKNGVIVVGHRESSVPFSYYDNQQKVVGYSQDYSNAIVEAVKKKLNKPDLQVKLIPITSQNRIPLLQNGTFDFECGSTTNNVERQKQAAFSDTIFVVGTRLLTKKGGDIKDFANLKDKAVVVTSGTTSEVLLNKLNEEQKMNMRIISAKDHGDSFRTLESGRAVAFMMDDVLLAGERAKAKKPDNWEIVGKPQSQEAYGCMLRKDDPQFKKLMDDTIAQVQTSGEAEKWFDKWFKNPILVSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSVLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKGGTGGSMSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKLICTTGKLPVPWPTLVTTLGYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNNPLNMNFELSDEMKALFKEPNDKALK*'

    dataColumns = [0,1,2,3,4,5,6,7,8] # these are plotted
    axisNames = list(repeat(' ', 9))
    # make it explicit! 
    axisNames[0] = ('Var#','Variant number')
    axisNames[1] = ('dF/F','fluorescence response')
    axisNames[2] = ('Kd','binding affinity')
    axisNames[3] = ('Kon','response rise time, 1/sec')
    axisNames[4] = ('dF1','change in F, 1 action potential')
    axisNames[5] = ('dk1','decay in F after 1 action potential')
    axisNames[6] = ('dF20','change in F after 20 action potentials')
    axisNames[7] = ('dk20','decay in F after 20 action potentials')
    axisNames[8] = ('Fitn','estimate of fitness based on dF/F and Kon')

    xaxis = 1
    yaxis = 3
    caxis = 2
else:
    fil = pd.read_excel('gcamp6variants.xlsx',sheet_name='Sheet1')

    wildTypeSeq = 'MGSHHHHHHGMASMTGGQQMGRDLYDDDDKDLATMVDSSRRKWNKTGHAVRAIGRLSSLENVYIKADKQKNGIKANFKIRHNIEDGGVQLAYHYQQNTPIGDGPVLLPDNHYLSVQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKGGTGGSMVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYIQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNTRDQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGDGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK'

    dataColumns = [0,1,2,3,4,5,6,7,8] # these are plotted
    axisNames = list(repeat(' ', 9))
    # make it explicit! 
    axisNames[0] = ('Var#','Variant number')
    axisNames[1] = ('dF1','change in F, 1 action potential')
    axisNames[2] = ('dF3','change in F, 3 action potentials')
    axisNames[3] = ('dF10','change in F, 10 action potentials')
    axisNames[4] = ('dF160','change in F, 160 action potentials')
    axisNames[5] = ('dk1','decay in F after 1 action potential')
    axisNames[6] = ('dk3','decay in F after 3 action potentials')
    axisNames[7] = ('dk10','decay in F after 10 action potentials')
    axisNames[8] = ('dk160','decay in F after 160 action potentials')

    xaxis = 3
    yaxis = 7
    caxis = 5


# convert to an array
wildType = []
wildType[:] = wildTypeSeq

prog = re.compile('\w\d+\w')
subprog = re.compile('(\w)(\d+)(\w)')

def fillFullSeq(fil):
    fullseq = {}
    for i,row in fil.iterrows():
        mu = fil['Mutations'][i]
        if type(mu) != type('hello'):
            mu = ' '
        seq = wildType.copy()
        museg = prog.findall(mu)
        # print(f'[{i}]', end=' ')
        for m in museg:
            # print(m, end= ' ')
            res = subprog.search(m)
            loc = int(res.group(2)) - 1
            if wildType[loc] != res.group(1):
                print(f'[{i+2}] mutation questionable: wildType {wildType[loc]} but {m} at {loc+1}')
            # well, apply the mutation. 
            seq[loc] = res.group(3)
            # print(f'updating {loc+1} to {res.group(3)}')
        seqstr = ''.join(seq) # back to string
        if Abhi: 
            vname = fil['Variant Name'][i]
            dff = fil['dF/F'][i]
            kd = fil['Kd (uM)'][i]
            kon = fil['Kon (stopped flow)'][i]
            dff1 = fil['1AP dFF'][i]
            dk1 = fil['1AP t1/2'][i]
            dff20 = fil['20AP dFF'][i]
            dk20 = fil['20AP t1/2'][i]
            if math.isnan(dff): 
                dff = 0.0
            if math.isnan(kd): 
                kd = 0.0
            if math.isnan(kon): 
                kon = 0.0
            if math.isnan(dff1): 
                dff1 = 0.0
            if math.isnan(dk1): 
                dk1 = 0.0
            if math.isnan(dff20): 
                dff20 = 0.0
            if math.isnan(dk20): 
                dk20 = 0.0
            varno = fil['Variant #'][i]
            fitness = 0.0 # placeholder; computed in screen for dups
            fullseq[i] = (seqstr, vname, [varno, dff, kd, kon, dff1, dk1, dff20, dk20, fitness])
        else:
            col = fil.columns
            vname = str(fil['variant'][i])
            df1 = fil[col[2]][i]
            df3 = fil[col[3]][i]
            df10 = fil[col[4]][i]
            df160 = fil[col[5]][i]
            dk1 = fil[col[6]][i]
            dk3 = fil[col[7]][i]
            dk10 = fil[col[8]][i]
            dk160 = fil[col[9]][i]
            fullseq[i] = (seqstr, vname, [i+2, df1, df3, df10, df160, dk1, dk3, dk10, dk160])

    # need to remove duplicates (same sequence) and average their measurements. 
    deleteMe = []
    for col in fullseq.keys():
        (seq,vname,dat) = fullseq[col]
        nvdat = np.zeros(len(dat))
        vdat = np.zeros(len(dat))
        for i in range(len(dat)):
            d = dat[i]
            if d > 0.0:
                vdat[i] = d
                nvdat[i] = nvdat[i] + 1
        for col2 in fullseq.keys():
            (seq2,vname2,dat2) = fullseq[col2]
            if col2 > col and seq == seq2:
                print(f'replicate found, [{col2+2}] {vname2} same as [{col+2}] {vname}')
                for i in range(len(dat2)):
                    d = dat2[i]
                    if d > 0.0 and i > 0: # don't average vno
                        vdat[i] = vdat[i] + d
                        nvdat[i] = nvdat[i] + 1
                deleteMe.append(col2)
                # null the sequence, so we don't count twice. 
                fullseq[col2] = (f'NULL-{col2}',vname2, dat2)
        for i in range(len(dat)):
            if nvdat[i] > 0:
                vdat[i] = vdat[i] / nvdat[i]
        if Abhi: 
            dff = vdat[1]
            kon = vdat[3]
            wildTypeFitness = 0.01430929425393453
            dffl = 2.0 / (1.0 + math.exp(dff / -30.0)) -1.0
            konl = 2.0 / (1.0 + math.exp(kon / -500.0)) -1.0
            fitness = dffl * konl / wildTypeFitness
            if col == 0:
                print(f'Wildtype fitness: {fitness}')
            vdat[8] = fitness
        fullseq[col] = (seq,vname,vdat) #update!
                
    for col in deleteMe:
        del fullseq[col]
        
    return fullseq

fullseq = fillFullSeq(fil) # prevents variable pollution

axisMin = np.zeros(max(dataColumns)+1) #indexed the same as the data columns
axisMin[0] = 1e5 #others, start from zero. 
axisMax = np.ones(max(dataColumns)+1) * -1e5
axisExp = np.ones(max(dataColumns)+1) # distory the axes for ease of visualization
for ax in dataColumns: 
    for col in fullseq.keys():
        (seq,vname,dat) = fullseq[col]
        d = dat[ax]
        axisMin[ax] = min(axisMin[ax], d)
        axisMax[ax] = max(axisMax[ax], d)

if not Abhi:
    axisExp[1] = 0.5
    axisExp[2] = 0.56
    axisExp[3] = 0.62
    axisExp[4] = 0.7
    axisExp[5] = 0.5
    axisExp[6] = 0.56
    axisExp[7] = 0.62
    axisExp[8] = 0.7
    
# want to plot the CDFs of the dependent variables, perhaps to make the data easier to visualize
def plotCDF(datacol):
    dat = []
    for key in fullseq.keys():
        (_,_,d) = fullseq[key]
        dat.append(d[datacol])
    dat = sorted(dat) # normal ascending sort
    y = torch.tensor(range(0, len(dat))) / (len(dat) - 1.0)
    plt.plot(dat, y.cpu().numpy())
    y2 = torch.pow(torch.tensor(dat) / max(dat), axisExp[datacol])
    plt.plot(dat, y2.cpu().numpy())
    plt.show()
    
# plotCDF(2)

def diffSeq(seq1, seq2):
        mucount = 0
        muname = ''
        for i in range(len(seq1)): 
            a = seq1[i]
            b = seq2[i]
            if a != b: 
                if mucount == 0:
                    muname = a + str(i+1) + b
                else:
                    muname = muname + ' ' + a + str(i+1) + b
                mucount = mucount + 1
        return (mucount, muname)

def doRegression(datacol, svgCol, fid, datacolName, zscoreThresh):
    # First make a list of all AA variants in each position, but only for sequences with valid data.
    global nShuf
    subList = []
    nSubs = 0
    for i in range(len(wildType)):
        st = {}
        for col in fullseq.keys():
            (seq,vname,dat) = fullseq[col]
            d = dat[datacol]
            aa = seq[i]
            if d > 0.0:
                if aa in st.keys():
                    st[aa] = st[aa] + 1
                else:
                    st[aa] = 1
        # print(f'[{i}] : {st}')
        if len(st) > 1:
            subList.append((i, st))
            nSubs = nSubs + len(st)
    # subList is a list of (i, st) where st is a list of substitutions and their counts. 
    # (omits positions where all AA are the same in all sequences)
    # print(subList)

    # need to flatten subList
    subListFlat = []
    for pos,st in subList:
        for aa in st.keys():
            cnt = st[aa]
            subListFlat.append((pos, aa, cnt)) # these are location (zero based), AA, counttuples.
    
    # get all entries where dff != 0.0
    nd = 0
    for col in fullseq.keys():
        (seq,vname,dat) = fullseq[col]
        d = dat[datacol]
        if d > 0.0:
            nd = nd + 1
            
    # need to make a binary vector indicating the presence or absence of a given amino acid for each sequence. 
    subMatrix = np.zeros((nSubs, nd))
    nnd = 0 # for building below
    dnp = np.zeros(nd) # dependent variables, numpy vector
    for col in fullseq.keys():
        (seq,vname,dat) = fullseq[col]
        d = dat[datacol]
        if d > 0.0:
            dnp[nnd] = d
            ns = 0
            for pos,st in subList:
                for sb in st:
                    if seq[pos] == sb:
                        subMatrix[ns, nnd] = 1
                    ns = ns+1
            nnd = nnd + 1
            
    # positive control
    nMW = math.floor(nShuf/18.0)
    positiveControl = False
    if positiveControl:
        subMatrix = np.random.rand(nSubs, nd) > 0.5 # binary, like the AA matrix
        oweights = np.random.randn(nSubs)
        dnp = np.matmul(np.transpose(subMatrix), oweights)
        nShuf = 1000
        nMW = 300

            
    #plt.imshow(subMatrix)
    #plt.show() # seems legit.
    # first predict dF/F
    subMatrix = np.transpose(subMatrix)
    if Abhi:
        fitPart = math.floor(nd * 0.8) # only some of the sequences
        subPart = math.floor(nSubs * 0.6) # only some of the substitutions
    else:
        fitPart = math.floor(nd * 0.8) # only some of the sequences
        subPart = math.floor(nSubs * 0.4) # only some of the substitutions
    meanWeights = np.zeros((nSubs, nMW))
    meanWeightsCount = np.zeros(nSubs, dtype=np.int32)
    for i in range(nShuf):
        idx = np.argsort(np.random.rand(nd))
        subidx = np.argsort(np.random.rand(nSubs))
        ssm1 = subMatrix[idx[:fitPart], :] # regularize by leaving both samples and AA subs out 
        ssm = ssm1[:, subidx[:subPart]]
        ssm = np.append(ssm, np.ones((fitPart,1)), 1) # DC term
        try:
            (weights,res,rank,singVal) = np.linalg.lstsq(ssm, dnp[idx[:fitPart]])
        except np.linalg.LinAlgError as e:
            print('SVD did not converge (or other error)')
            weights[0] = 1e9
        if max(abs(weights)) < 1e6:
            print(f'mean dnp: {np.mean(dnp[idx[:fitPart]])}')
            ssm = np.append(subMatrix[:, subidx[:subPart]], np.ones((nd, 1)), 1) # DC term
            pred = np.matmul(ssm, weights)
            # now, trick is cross-validation .. subMatrix is poorly conditioned (278 x 229)
            #plt.scatter(dFFnp, pred)
            #plt.xlabel('real dF/F')
            #plt.ylabel('predicted dF/F')
            #plt.show()
            for j in range(subPart):
                src = subidx[j]
                mwc = meanWeightsCount[src]
                if mwc < nMW:
                    meanWeights[src, mwc] = weights[j]
                    meanWeightsCount[src] = mwc + 1
            
    #plt.plot(meanWeightsCount)
    #plt.show()

    meanM = np.mean(meanWeights, 1)
    stdM = np.std(meanWeights, 1)
    zscore = meanM/stdM
    idx = np.argsort(zscore, kind='stable') # idx indexes zscore - sorts in ascending order
    idx = np.flip(idx)
    if not positiveControl:
        if zscoreThresh < 1.0:
            idx = idx[ abs(zscore[idx]) > zscoreThresh]
        else:
            # take the top-N of the list. 
            idx2 = np.argsort(abs(zscore[idx]), kind='stable')
            idx2 = np.flip(idx2) # descending order
            idx = idx[idx2[0:math.floor(zscoreThresh)]]
            # this will permute, alas
            idx3 = np.argsort(zscore[idx])
            idx3 = np.flip(idx3)
            idx = idx[idx3]
            # slight mindfuck
    #plt.errorbar(range(len(idx)), meanM[idx], yerr=stdM[idx], fmt='o')
    #if positiveControl:
        #plt.plot(range(len(idx)), oweights[idx], 'o', color='red')
    #for i in range(len(idx)):
        #id = idx[i]
        #(loc, aa, cnt) = subListFlat[id]
        #print(f'{loc+1}{aa}: {zscore[id]:.3f} ({cnt} of {nd})')
    #print('\n')
    #plt.show()
    
    
    pathAxes = '''<path style="fill:none;stroke:#000000;stroke-width:0.1px;stroke-linecap:round;stroke-linejoin:miter;stroke-opacity:1.0"\n'''
    fid.write(pathAxes)
    fid.write(f'd="M {svgCol},{0.5} {svgCol},{len(idx)+1}"\n')
    fid.write('/>\n') # endpath
    textAxes = '''<text 
style="font-style:normal;font-weight:normal;font-size:2px;line-height:1.25;font-family:serif;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.26458332"\n'''
    fid.write(textAxes)
    fid.write(f'x="{svgCol-2}" y="{len(idx)+3}">{datacolName}</text>\n')
    fid.write('/>\n') # endtext
    
    # now need to shoehorn this into a SVG. 
    pathWhisker = '''<path style="fill:none;stroke:#0000e0;stroke-width:0.18px;stroke-linecap:round;stroke-linejoin:miter;stroke-opacity:0.75"\n'''
    textHead = '<text style="font-size:1px; font-family:sans-serif;" \n'
    
    mnscl = 3.0 / max(abs(meanM))
    
    def lineWhisker(fid, y, mean, std, name, cnt):
        fid.write('<g>\n')
        fid.write(pathWhisker)
        x = mean*mnscl + svgCol
        fid.write(f'd="M {x-mnscl*std},{y} {x+mnscl*std},{y}"\n')
        fid.write('/>\n') # endpath
        fid.write(f'<ellipse id="effect{name}"\n')
        fid.write(f'style="fill:#0000e0;fill-opacity:0.45;stroke:None" \n')
        fid.write(f'cx="{x}"\n')
        fid.write(f'cy="{y}"\n')
        fid.write('rx="0.6"\n')
        fid.write('ry="0.6"\n')
        fid.write('onclick="changeSubs(this.id)"\n')
        fid.write('onmouseenter="showLabel(this.id)"\n')
        fid.write('onmouseleave="hideLabel(this.id)"\n')
        fid.write('/>\n') # end the ellipse
        fid.write(f'<ellipse id="effect{name}_enabled"\n')
        fid.write(f'style="fill:None;fill-opacity:0.45;stroke:#ff0000;stroke-width:0.125;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:0.8" \n')
        fid.write(f'cx="{x}"\n')
        fid.write(f'cy="{y}"\n')
        fid.write('rx="0.6"\n')
        fid.write('ry="0.6"\n')
        fid.write('visibility="hidden"\n')
        fid.write('/>\n') # end the 'enabled' ellipse
        fid.write(textHead)
        fid.write(f'x="{svgCol+6}" y="{y+0.5}">{name} ({cnt})</text>\n')
        fid.write('</g>\n')
        
    def lineWhiskerLabel(fid, y, mean, std, name, nameLong):
        # roll-over label
        x = mean*mnscl + svgCol
        yp = 0.75
        fid.write('<g>\n')
        for lbl in nameLong:
            fid.write(textHead + f'id="effect{name}_label"\n')
            fid.write('visibility="hidden"\n')
            fid.write(f'x="{x+0.75}" y="{y+yp}">{lbl}</text>\n')
            yp = yp + 1.0
        fid.write('</g>\n')
    
    for paz in range(2):
        for i in range(len(idx)):
            id = idx[i]
            mean = meanM[id]
            std = stdM[id]
            (loc, aa, cnt) = subListFlat[id]
            name = f'{loc+1}{aa}'
            if paz == 0 : 
                lineWhisker(fid, i+1, mean, std, name, cnt)
            else:
                nameLong = [f'{loc+1}{aa}: {cnt} of {nd}',f'zscore:{zscore[id]:.3f}',f'mean:{mean:.2f} +-{std:.2f}']
                lineWhiskerLabel(fid, i+1, mean, std, name, nameLong)
                
def doDiffRegression(datacol, svgCol, fid, datacolName, displayLabel, labelStr):
    mulist = {} # dictionary of lists, each with a delta dat value in it
    for key in fullseq.keys():
        (seq,_,dat) = fullseq[key]
        for key2 in fullseq.keys():
            if key2 >= key:
                (seq2,_,dat2) = fullseq[key2]
                (mucount, muname) = diffSeq(seq, seq2)
                if mucount == 1:
                    d = dat2[datacol] - dat[datacol]
                    if muname in mulist.keys():
                        lst = mulist[muname]
                        lst.append(d)
                        mulist[muname] = lst
                    else:
                        # reverse it and see if that's in the list
                        (mucount2, muname2) = diffSeq(seq2, seq)
                        if muname2 in mulist.keys():
                            lst = mulist[muname2]
                            lst.append(-1*d)
                            mulist[muname2] = lst
                        else:
                            mulist[muname] = [d]

    # sort the dictionary by length of delta list
    mulist2 = sorted(mulist.items(), key = lambda v:(len(v[1])) , reverse=True)
    cnt = 0
    nmu = 80
    #for mu in mulist2:
        #if cnt < nmu:
            #print(mu)
        #cnt = cnt+1
    print(f'Length mulist: {len(mulist)}')
    
    pathAxes = '''<path style="fill:none;stroke:#000000;stroke-width:0.1px;stroke-linecap:round;stroke-linejoin:miter;stroke-opacity:1.0"\n'''
    fid.write(pathAxes)
    fid.write(f'd="M {svgCol},{0.5} {svgCol},{1.1*nmu}"\n')
    fid.write('/>\n') # endpath
    textAxes = '''<text 
style="font-style:normal;font-weight:normal;font-size:2px;line-height:1.25;font-family:serif;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.26458332"\n'''
    fid.write(textAxes)
    fid.write(f'x="{svgCol-2}" y="{1.1*nmu+3}">{datacolName}</text>\n')
    fid.write('/>\n') # endtext
    
    # rather than making a whisker-plot, as with abhi, we can plot the effect of all the mutations, 
    # with a central circle for mean.
    maxdelta = 0
    cnt = 0
    for mu in mulist2:
        if cnt < nmu:
            dat = mu[1]
            mxdat = np.max(np.abs(dat))
            maxdelta = max([maxdelta, mxdat])
        cnt = cnt+1
    cnt = 0
    mnscl = 4.0 / maxdelta
    for mu in mulist2:
        if cnt < nmu:
            muname = mu[0]
            dat = mu[1]
            y = 1.1*cnt + 1
            fid.write('<g>\n')
            for d in dat: 
                x = d*mnscl + svgCol
                fid.write(f'<ellipse \n')
                fid.write(f'style="fill:#5000e0;fill-opacity:0.25;stroke:None" \n')
                fid.write(f'cx="{x}"\n')
                fid.write(f'cy="{y}"\n')
                fid.write('rx="0.2"\n')
                fid.write('ry="0.2"\n')
                fid.write('/>\n') # end the data ellipse
            mean = np.mean(dat)
            std = np.std(dat)
            x = mean*mnscl + svgCol
            shortmuname = muname[1:]
            fid.write(f'<ellipse id="effect{shortmuname}"\n')
            fid.write(f'style="fill:#0000e0;fill-opacity:0.45;stroke:None" \n')
            fid.write(f'cx="{x}"\n')
            fid.write(f'cy="{y}"\n')
            fid.write('rx="0.6"\n')
            fid.write('ry="0.6"\n')
            fid.write('onclick="changeSubs(this.id)"\n')
            fid.write(f'onmouseenter="showLabel(this.id)"\n')
            fid.write(f'onmouseleave="hideLabel(this.id)"\n')
            fid.write('/>\n') # end the ellipse
            fid.write(f'<ellipse id="effect{shortmuname}_enabled"\n')
            fid.write(f'style="fill:None;fill-opacity:0.45;stroke:#ff0000;stroke-width:0.125;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:0.8" \n')
            fid.write(f'cx="{x}"\n')
            fid.write(f'cy="{y}"\n')
            fid.write('rx="0.6"\n')
            fid.write('ry="0.6"\n')
            fid.write('visibility="hidden"\n')
            fid.write('/>\n') # end the 'enabled' ellipse
            # visible text
            if displayLabel:
                fid.write(textHead) 
                fid.write(f'x="{svgCol+6}" y="{y+0.5}">{muname} ({len(dat)})</text>\n')
            # rolover text (slightly different -- ignore the original residue)
            longLabels = [f'{muname} (n={len(dat)})', f'{mean:.2f} +-{std:.2f}']
            ii = 0
            for labl in longLabels:
                labelStr = labelStr + textHead + f'id="effect{shortmuname}_label"\n' # this allows for cross-referencing with the edge graph etc
                labelStr = labelStr + 'visibility="hidden"\n'
                labelStr = labelStr + f'x="{x+0.75}" y="{y+0.75+ii}">{labl}</text>\n'
                ii = ii + 1
            fid.write('</g>\n')
        cnt = cnt+1
    return labelStr

if Abhi:
    fid = open('sniffer.html', 'w')
else:
    fid = open('gcamp.html', 'w')

fid.write('''<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>\n''')
if Abhi:
    pageName = 'GluSnFR'
else:
    pageName = 'GCaMP'
fid.write(f'<title>{pageName} interactive analysis</title>\n')
fid.write('''<script src="sniffer.js"></script>
<style>
  .container {
    position: relative;
    display: inline-block; /* shrink wrap */
    height: 98vh; /* arbitrary input height; adjust this; good at 95 */
    margin-left: 10px;
  }
  .container > img {
    height: 100%; /* make the img's width 100% of the height of .container */
  }
  .contents {
    /* match size of .container, without influencing it */
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
  }
</style>
</head>
<body>\n''')

textHead = '<text style="font-size:1px; font-family:sans-serif;" \n'
textHeadSerif = '<text style="font-size:1px; font-family:serif;" \n'

pathHead = '''<path style="fill:none;stroke:#000000;stroke-width:0.38px;stroke-linecap:round;stroke-opacity:0.24;marker-mid:url(#Arrow1Lend1)"\n'''
pathHead2 = '''<path style="fill:none;stroke:#00a0a0;stroke-width:0.38px;stroke-linecap:round;stroke-opacity:0.23;marker-mid:url(#Arrow1Lend2)"\n'''
pathHead3 = '''<path style="fill:none;stroke:#a000a0;stroke-width:0.38px;stroke-linecap:round;stroke-opacity:0.17;marker-mid:url(#Arrow1Lend3)"\n'''



fid.write('''
<div class="container">
  <!-- transparent image with 1:1 intrinsic aspect ratio -->
  <img src="bg_16x9.png">
  <div class="contents">\n''' )
fid.write('<svg class="pat" x="0px" y="0px" viewBox="0 0 177.7 100.0" style="enable-background:new 0 0 177.7 100.0;">\n')
fid.write('''<defs
     id="defs1138">
    <marker
       orient="auto"
       refY="0.0"
       refX="0.0"
       id="Arrow1Lend1"
       style="overflow:visible;">
      <path
         id="path829"
         d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z "
         style="fill-rule:evenodd;stroke:#202020;stroke-width:0.1pt;stroke-opacity:0;fill:#000000;fill-opacity:0.18"
         transform="scale(0.4) rotate(180) translate(12.5,0)" />
    </marker>
    <marker
       orient="auto"
       refY="0.0"
       refX="0.0"
       id="Arrow1Lend2"
       style="overflow:visible;">
      <path
         id="path829"
         d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z "
         style="fill-rule:evenodd;stroke:#000000;stroke-width:0.1pt;stroke-opacity:0;fill:#00a0a0;fill-opacity:0.2"
         transform="scale(0.4) rotate(180) translate(12.5,0)" />
    </marker>
    <marker
       orient="auto"
       refY="0.0"
       refX="0.0"
       id="Arrow1Lend3"
       style="overflow:visible;">
      <path
         id="path829"
         d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z "
         style="fill-rule:evenodd;stroke:#000000;stroke-width:0.1pt;stroke-opacity:0;fill:#a000a0;fill-opacity:0.14"
         transform="scale(0.4) rotate(180) translate(12.5,0)" />
    </marker>
  </defs>''') # for a mid-point direction arrow
labelStr = ''
if Abhi:
    doRegression(1, 108.0, fid, 'dF/F', 0.65)
    doRegression(2, 126.0, fid, 'Kd (um)', 0.5)
    doRegression(3, 144.0, fid, 'Kon', 0.6)
    doRegression(8, 162.0, fid, 'Fitness', 0.6)
else: # doesn't work! 
    labelStr = doDiffRegression(1, 108.0, fid, 'dF/F 1', False, labelStr)
    labelStr = doDiffRegression(3, 118.0, fid, 'dF/F 10', False, labelStr)
    labelStr = doDiffRegression(4, 128.0, fid, 'dF/F 160', False, labelStr)
    labelStr = doDiffRegression(5, 138.0, fid, 'Decay 1', False, labelStr)
    labelStr = doDiffRegression(7, 148.0, fid, 'Decay 10', False, labelStr)
    labelStr = doDiffRegression(8, 158.0, fid, 'Decay 160', True, labelStr)
    #doRegression(1, 108.0, fid, 'dF/F 1FP', 50)
    #doRegression(3, 126.0, fid, 'dF/F 10FP', 50)
    #doRegression(5, 144.0, fid, 'Decay 1FP', 50)
    #doRegression(7, 162.0, fid, 'Decay 10FP', 50)

# alright, need to make the plotting capabilities generic -- you can plot anything against anything else
# in this case, there are only three variables. 
# in the case of GCaMP*, there are a great many more variables, so it makes more sense. 
# Idea is to add in data_ fields for each of the axes on each of the elements that needs to move. 
# javascript will access this data when adjusting element positions.
# overall data to screen transform is otherwise the same. 
# Axes labels are auto-generated and shown/hidden based on selection. 
# no animations. 
# 

def realToScreenX(xin, axis):
    norm = (xin-axisMin[axis])/(axisMax[axis]-axisMin[axis])
    x = 90.0*math.pow(norm, axisExp[axis])+8.0
    return x
    
def realToScreenY(yin, axis):
    norm = (yin-axisMin[axis])/(axisMax[axis]-axisMin[axis])
    y = 100.0 - (90.0*math.pow(norm, axisExp[axis])+8.0)
    return y

def variantToLocs(key):
    # returns a tuple of two arrays, X and Y, each indexed by axis
    (seq,_,dat) = fullseq[key]
    xl = np.zeros(axisMin.shape)
    yl = np.zeros(axisMin.shape)
    for ax in dataColumns:
        xl[ax] = realToScreenX(dat[ax], ax)
        yl[ax] = realToScreenY(dat[ax], ax)
    return (xl, yl)

locs = {key: variantToLocs(key) for key in fullseq.keys() }

def makeMuPaths(fullseq, locs, xaxis, yaxis, nmu, labelStr):
    grStr = ''
    for key in fullseq.keys():
        (seq,_,_) = fullseq[key]
        (xl,yl) = locs[key]
        x = xl[xaxis]
        y = yl[yaxis]
        muno = 0 # this is for associating label and line -> hover-over visibility
        for key2 in fullseq.keys():
            if key != key2:
                (seq2,_,_) = fullseq[key2]
                # i think the proper way to do this is simply iterate over the list
                (mucount, muname) = diffSeq(seq, seq2)
                if mucount > 0 and mucount < nmu+1:
                    (xl2,yl2) = locs[key2]
                    x2 = xl2[xaxis]
                    y2 = yl2[yaxis]
                    if mucount == 1:
                        grStr = grStr + pathHead
                    if mucount == 2:
                        grStr = grStr + pathHead2
                    if mucount >= 3:
                        grStr = grStr + pathHead3
                    grStr = grStr + f'd="M {x},{y} {(x+x2)/2.0},{(y+y2)/2.0} {x2},{y2}"\n'
                    grStr = grStr + f'id="variant{key}_1mu" \nvisibility="hidden"\n'
                    grStr = grStr + f'data-mutationtag="{muname}" \n'
                    grStr = grStr + f'data-muno="{muno}" \n'
                    grStr = grStr + f'onclick="showMutation(this.id, this.dataset.mutationtag)"\n'
                    grStr = grStr + f'onmouseenter="showLabel2(this.id, {muno}, this.dataset.mutationtag)"\n'
                    grStr = grStr + f'onmouseleave="hideLabel2(this.id, {muno}, this.dataset.mutationtag)"\n'
                    for axis in dataColumns:
                        grStr = grStr + f'data-xone{axis}="{xl[axis]}"\n'
                        grStr = grStr + f'data-yone{axis}="{yl[axis]}"\n'
                        grStr = grStr + f'data-xtwo{axis}="{xl2[axis]}"\n'
                        grStr = grStr + f'data-ytwo{axis}="{yl2[axis]}"\n'
                        # will need to interpolate these to move the mid-point arrow. 
                        # don't need to duplicate the data for the text tag; move it in JS at the same time.
                    grStr = grStr + '/>\n'
                    
                    labelStr = labelStr + textHead
                    labelStr = labelStr + 'visibility="hidden"\n'
                    labelStr = labelStr + f'id="variant{key}_1mu_label{muno}" \n'
                    labelStr = labelStr + f'tag_mutation="{muname}" \n'
                    labelStr = labelStr + f'x="{(x+x2)/2.0}" y="{(y+y2)/2.0-0.25}">{muname}</text>\n'
                    
                    muno = muno +1
    return (grStr, labelStr)

# makeMuPaths is too heavy for the GCaMP data -- SSM leaves the graph full. 
# instead, let's make a type of minimum-spanning-tree, 
# though with greedy search from the starting sequence, adding nodes based on which ones have the largest 1-mu neighbors
def makeMuMST(fullseq, locs, xaxis, yaxis, labelStr):
    grStr = ''
    numSingleMu = {} # dictionary containing the number of sequences one mutation away from the keyed seq
    for key in fullseq.keys():
        (seq,_,_) = fullseq[key]
        nsm = 0
        for key2 in fullseq.keys():
            if key2 != key:
                (seq2,_,_,) = fullseq[key2]
                (mucount, _) = diffSeq(seq, seq2)
                if mucount == 1:
                    nsm = nsm + 1
        numSingleMu[key] = nsm
        
        
    # ok, now iterate over the list of variants, adding greedily to the graph. 
    fullEdges = [] # list of all edges in the graph
    keyEdges = {} # list of edge indexes (above), with the same key as fullseq
    for key in fullseq.keys(): 
        (seq,vname,_) = fullseq[key]
        #bestmucount = 600
        #bestnsm = 600
        #bestkey = key
        #bestmuname = ''
        #for key2 in fullseq.keys().__reversed__(): # go backwards to facilitate watermark algo
            #if key2 != key:
                #(seq2,_,_,) = fullseq[key2]
                #(mucount, muname) = diffSeq(seq, seq2)
                #nsm = numSingleMu[key2]
                #if mucount < bestmucount:
                    #bestmucount = mucount
                    #bestnsm = nsm
                    #bestkey = key2
                    #bestmuname = muname
                #if nsm < bestnsm and mucount <= bestmucount:
                    #bestmucount = mucount
                    #bestnsm = nsm
                    #bestkey = key2
                    #bestmuname = muname
                #if key2 < bestkey and nsm <= bestnsm and mucount <= bestmucount:
                    #bestmucount = mucount
                    #bestnsm = nsm
                    #bestkey = key2
                    #bestmuname = muname
                ## this might be useful for ranking the nodes... yes, later.
                    
        # beh, that didn't do very much.  How about lineage? 
        (gcamp6sSeq,_,_) = fullseq[0]
        (gcamp6fSeq,_,_) = fullseq[1]
        (diffSlow,muSlow) = diffSeq(gcamp6sSeq, seq) # format: original position new
        (diffFast,muFast) = diffSeq(gcamp6fSeq, seq)
        if diffFast < diffSlow:
            refSeq = gcamp6fSeq
            refMu = muFast
        else:
            refSeq = gcamp6sSeq
            refMu = muSlow
        # for each of the single AA substitutions, there are (potentially) a lot of variants that match. 
        # as before, choose these greedily based on nsm and key. 
        muList = refMu.split(' ')
        (xl,yl) = locs[key]
        x = xl[xaxis]
        y = yl[yaxis]
        seq1 = seq
        key1 = key
        edges = [] # list of edges in this lineage.
        while len(muList) > 0 and len(refMu) > 0:
            # print(f'muList: {muList}')
            bestmucount = 600
            bestnsm = 600
            bestkey = key1
            bestmuname = ''
            for key2 in fullseq.keys().__reversed__(): # go backwards to facilitate watermark algo
                if key2 != key1:
                    (seq2,_,_,) = fullseq[key2]
                    nsm = numSingleMu[key2]
                    (mucount, muname) = diffSeq(seq2, seq1) # forward chronology naming
                    overlap = True
                    if mucount >= 1:
                        for mu in muname.split(' '):
                            if mu not in muList:
                                overlap = False
                    else:
                        overlap = False
                    if overlap: 
                        if mucount < bestmucount:
                            bestmucount = mucount
                            bestnsm = nsm
                            bestkey = key2
                            bestmuname = muname
                        if nsm < bestnsm and mucount <= bestmucount:
                            bestmucount = mucount
                            bestnsm = nsm
                            bestkey = key2
                            bestmuname = muname
                        if key2 < bestkey and nsm <= bestnsm and mucount <= bestmucount:
                            bestmucount = mucount
                            bestnsm = nsm
                            bestkey = key2
                            bestmuname = muname
                
            # print(f'for {vname} on [{key1+2}] bestkey is [{bestkey+2}] with mucount:{bestmucount} nsm:{bestnsm} {bestmuname}')
            newEntry = (key1, bestkey, bestmucount, bestmuname)
            if newEntry in fullEdges:
                edgeNo = fullEdges.index(newEntry)
            else:
                edgeNo = len(fullEdges)
                fullEdges.append(newEntry)
            edges.append(edgeNo)
        
            (seq1,_,_) = fullseq[bestkey]
            key1 = bestkey
            for mu in bestmuname.split(' '):
                if mu in muList:
                    muList.remove(mu)
            if key1 == 0 or key1 == 1:
                muList = []
        # end while
        keyEdges[key] = edges
    #end loop over keys
    
    # write all edges to svg
    muno = 0
    for edge in fullEdges:
        (key1, key2, mucount, muname) = edge
        (xl,yl) = locs[key1]
        (xl2,yl2) = locs[key2]
        x = xl[xaxis]
        y = yl[yaxis]
        x2 = xl2[xaxis]
        y2 = yl2[yaxis]
        if mucount == 1:
            grStr = grStr + pathHead
        if mucount == 2:
            grStr = grStr + pathHead2
        if mucount >= 3:
            grStr = grStr + pathHead3
        grStr = grStr + f'd="M {x},{y} {(x+x2)/2.0},{(y+y2)/2.0} {x2},{y2}"\n'
                        # this is ignored, updated in JS later
        grStr = grStr + f'id="lineage{muno}" \nvisibility="hidden"\n'
        grStr = grStr + f'data-mutationtag="{muname}" \n'
        grStr = grStr + f'data-muno="0" \n'
        grStr = grStr + f'onclick="showMutation(this.id, this.dataset.mutationtag)"\n'
        grStr = grStr + f'onmouseenter="showLabel2(this.id, 0, this.dataset.mutationtag)"\n'
        grStr = grStr + f'onmouseleave="hideLabel2(this.id, 0, this.dataset.mutationtag)"\n'
        for axis in dataColumns:
            grStr = grStr + f'data-xone{axis}="{xl2[axis]}"\n'
            grStr = grStr + f'data-yone{axis}="{yl2[axis]}"\n'
            grStr = grStr + f'data-xtwo{axis}="{xl[axis]}"\n'
            grStr = grStr + f'data-ytwo{axis}="{yl[axis]}"\n'
            # will need to interpolate these to move the mid-point arrow. 
            # don't need to duplicate the data for the text tag; move it in JS at the same time.
        grStr = grStr + '/>\n'
        labelStr = labelStr + textHead
        labelStr = labelStr + 'visibility="hidden"\n'
        labelStr = labelStr + f'id="lineage{muno}_label0" \n'
        labelStr = labelStr + f'tag_mutation="{muname}" \n'
        labelStr = labelStr + f'x="{(x+x2)/2.0}" y="{(y+y2)/2.0-0.75}">{muname}</text>\n'
        muno = muno +1
    
    # the javascript for turning this on and off needs to be in the ellipses
    return (grStr, labelStr, keyEdges)

def makeVarEllipses(fullseq, locs, xaxis, yaxis, caxis, labelStr, keyEdges):  
    for key in fullseq.keys():
        (seq,vname,dat) = fullseq[key]
        (xl,yl) = locs[key]
        varno = dat[0]
        x = xl[xaxis]
        y = yl[yaxis]
        if Abhi:
            col = (xl[caxis]-8.0)/40.0 # high-Kd variants saturate to red
        else:
            col = (xl[caxis]-8.0)/90.0 # normal mapping
        col = np.clip(col, 0.0, 1.0)
        color = matplotlib.colors.to_hex((col, 0.2, 1.0-col))
        alpha = 0.45
        if Abhi:
            if re.search('wild', vname, re.IGNORECASE):
                color = '#000000'
                alpha = 0.8
        else:
            if re.search('camp', vname, re.IGNORECASE):
                color = '#00ad00'
                alpha = 0.95
        fid.write('<g>\n')
        fid.write(f'<ellipse id="variant{key}"\n')
        fid.write(f'style="fill:{color};fill-opacity:{alpha};stroke:None;" \n')
        fid.write(f'cx="{x}"\n')
        fid.write(f'cy="{y}"\n')
        fid.write('rx="0.62"\n')
        fid.write('ry="0.62"\n')
        if Abhi:
            fid.write('onclick="changeLinks(this.id)"\n')
        else:
            fid.write(f'onclick="changeLineage(this.id,{keyEdges[key]})"\n')
        fid.write('onmouseenter="showLabel(this.id)"\n')
        fid.write('onmouseleave="hideLabel(this.id)"\n')
        for axis in dataColumns:
            fid.write(f'data-xone{axis}="{xl[axis]}"\n')
            fid.write(f'data-yone{axis}="{yl[axis]}"\n')
            # again, don't duplicate this data -- instead move the label and sequence ellipse in JS
        fid.write('/>\n') # end the ellipse
        
        if Abhi:
            kd = dat[2]
            kdstr = f'({kd} um)'
        else:
            kdstr = ''
        labelStr = labelStr + textHead + f'id="variant{key}_label"\n'
        labelStr = labelStr + 'visibility="hidden"\n'
        labelStr = labelStr + f'x="{x}" y="{y-0.95}">{vname} [{math.floor(varno)}] {kdstr}</text>\n'
        
        fid.write(f'<ellipse id="variant{key}"\n')
        fid.write(f'style="fill:None;fill-opacity:0.45;stroke:#ff0000;stroke-width:0.125;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:0.8" \n')
        fid.write(f'cx="{x}"\n')
        fid.write(f'cy="{y}"\n')
        fid.write('rx="0.62"\n')
        fid.write('ry="0.62"\n')
        fid.write(f'data-sequencetag="{seq}" \n')
        fid.write('visibility="hidden"\n')
        fid.write('/>\n') # end the open ellipse
        fid.write('</g>\n') # end ellipse group. 
    return labelStr
                
if Abhi:
    (grStr, labelStr) = makeMuPaths(fullseq, locs, xaxis, yaxis, 2, labelStr)
    # put here, since the data is much less dense.. 
    fid.write('<g>\n') # edges group! 
    fid.write(grStr)
    fid.write('</g>\n')
    keyEdges = {}
else:
    (grStr, labelStr, keyEdges) = makeMuMST(fullseq, locs, xaxis, yaxis, labelStr)
    
labelStr = makeVarEllipses(fullseq, locs, xaxis, yaxis, caxis, labelStr, keyEdges)

if not Abhi:
    fid.write('<g>\n') # edges group! 
    fid.write(grStr)
    fid.write('</g>\n')

fid.write('<g>\n') # label group! 
fid.write(labelStr)
fid.write('</g>\n')

# obviously, axes have to be programmatic.
pathAxes = '''<path style="fill:none;stroke:#000000;stroke-width:0.2px;stroke-linecap:round;stroke-opacity:1.0"\n'''
textAxes = '''<text 
style="font-style:normal;font-weight:normal;font-size:2px;line-height:1.25;font-family:serif;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1.0"\n'''
textHeadCenter = '<text style="font-size:1px; font-family:sans-serif;text-anchor:middle;text-align:center" \n'

def lineSC(fid, x1,y1,x2,y2):
    # input is screen coordinates.
    fid.write(pathAxes)
    fid.write(f'd="M {x1},{y1} {x2},{y2}"\n')
    fid.write('/>\n') # endpath

def makeAxis(fid, axis):
    print(f'makeAxis {axis}')
    mn = axisMin[axis]
    mx = axisMax[axis]
    # assume the min is > 0
    spread = mx - mn
    xp = 1 # exponent
    man = 1 # mantissa. so, 10
    done = False
    n = 50
    while not done and n > 0:
        n = n-1
        inc = math.pow(10, xp)*man
        sta = math.ceil(mn/inc)
        fin = math.floor(mx/inc)
        count = fin - sta + 1
        if count > 10:
            if man == 1:
                man = 2
            elif man == 2:
                man = 5
            elif man == 5:
                man = 1
                xp = xp + 1
        elif count < 5:
            if man == 1:
                man = 5
                xp = xp - 1
            elif man == 2:
                man = 1
            elif man == 5:
                man = 2
        else:
            done = True
        # print(f'min {mn} max {mx} inc {inc} count {count}') 
    sta = math.ceil(mn/inc)*inc
    fin = math.floor(mx/inc)*inc
    # ok! now we need to draw the lines and labels.  
    fid.write(f'<g id="axisX{axis}"\n')
    fid.write(f'visibility="hidden">\n')
    lineSC(fid, realToScreenX(mn, axis), 93.0, realToScreenX(mx, axis), 93.0)
    # assume there is a zero tick already..
    for xx in np.arange(sta, fin+0.1*inc, inc):
        xs = realToScreenX(xx, axis)
        lineSC(fid, xs, 93.0, xs, 94.0)
        fid.write(textHeadCenter)
        if inc < 0.1:
            lbl = f'{math.floor(xx*100.0)/100.0}'
        elif inc < 1.0:
            lbl = f'{math.floor(xx*10.0)/10.0}'
        else:
            lbl = f'{math.floor(xx)}'
        fid.write(f'x="{xs}" y="95.2">{lbl}</text>\n')
    fid.write('</g>\n')
    # add in a label. Not in the same group!
    fid.write(f'<g id="axisXlabel{axis}"\n')
    fid.write('style="opacity:0.4">\n')
    fid.write('<rect style="fill:#e0e0d0;fill-opacity:0.8;stroke:None"\n')
    fid.write('width="7"\n')
    fid.write('height="2.2"\n')
    fid.write(f'x="{50.0-32.0 + axis*8-1.1}"\n')
    fid.write(f'y="{98.0-2.1}"\n')
    fid.write(f'onclick="enableXAxis({axis})"\n')
    fid.write('/>\n') # end rect
    fid.write(textAxes)
    fid.write(f'id=XAxis{axis}\n')
    fid.write(f'onclick="enableXAxis({axis})"\n')
    fid.write('onmouseenter="showLabel(this.id)"\n')
    fid.write('onmouseleave="hideLabel(this.id)"\n')
    fid.write(f'x="{50.0-32.0 + axis*8}" y="97.7">{axisNames[axis][0]}</text>\n')
    fid.write(textHead)
    fid.write(f'id=XAxis{axis}_label\n')
    fid.write('visibility="hidden"\n')
    fid.write(f'x="{50.0-32.0 + axis*8}" y="{97.7-2}">{axisNames[axis][1]}</text>\n')
    fid.write('</g>\n')
    
    # do the same thing only vertical
    fid.write(f'<g id="axisY{axis}"\n')
    fid.write(f'visibility="hidden">\n')
    lineSC(fid, 7.0, realToScreenY(mn, axis), 7.0, realToScreenY(mx, axis))
    for yy in np.arange(sta, fin+0.1*inc, inc):
        ys = realToScreenY(yy, axis)
        lineSC(fid, 6.0, ys, 7.0, ys)
        fid.write(textHead)
        if inc < 0.1:
            lbl = f'{math.floor(yy*100.0)/100.0}'
        elif inc < 1.0:
            lbl = f'{math.floor(yy*10.0)/10.0}'
        else:
            lbl = f'{math.floor(yy)}'
        fid.write(f'x="4.0" y="{ys-0.15}">{lbl}</text>\n')
    fid.write('</g>\n')
    # add in a label. Not in the same group!
    fid.write(f'<g id="axisYlabel{axis}"\n')
    fid.write('style="opacity:0.4">\n')
    fid.write('<rect style="fill:#e0e0d0;fill-opacity:0.8;stroke:None"\n')
    fid.write('width="5"\n')
    fid.write('height="3.5"\n')
    fid.write('x="0.0"\n')
    fid.write(f'y="{50.0-12.0 + axis*4 - 2.5}"\n')
    fid.write(f'onclick="enableYAxis({axis})"\n')
    fid.write('/>\n') # end rect
    fid.write(textAxes)
    fid.write(f'id=YAxis{axis}\n')
    fid.write(f'onclick="enableYAxis({axis})"\n')
    fid.write('onmouseenter="showLabel(this.id)"\n')
    fid.write('onmouseleave="hideLabel(this.id)"\n')
    fid.write(f'x="0.1" y="{50.0-12.0 + axis*4}">{axisNames[axis][0]}</text>\n')
    fid.write(textHead)
    fid.write(f'id=YAxis{axis}_label\n')
    fid.write('visibility="hidden"\n')
    fid.write(f'x="0.1" y="{50.0-14.5 + axis*4}">{axisNames[axis][1]}</text>\n')
    fid.write('</g>\n')

for ax in dataColumns:
    makeAxis(fid, ax)


fid.write(textHead)
if Abhi:
    fid.write(f'x="{50}" y="{1}">Color -> Kd</text>\n')
else:
    fid.write(f'x="{50}" y="{1}">Color -> Decay 1AP</text>\n')
fid.write(textHead)
fid.write(f'x="{50}" y="{2.5}">Click on mutations (edges) to toggle identical ones w/ other bg</text>\n')
fid.write(textHead)
fid.write(f'x="{50}" y="{4}">If things get confusing, reload (F5)</text>\n')

if Abhi:
    fid.write('''<text style="font-size:1px; font-family:serif;text-anchor:end;text-align:end;writing-mode:lr;"\n 
    x="161.5" y="99">\nData / protein engineering / etc by Abhi Aggarwal and Kaspar Podgorski.  Programming by Tim Hanson.  Artwork by Filip Tomaska.</text>\n''')
    
    fid.write('<image x="122" y="66" width="40" height="30" xlink:href="filip_tattoo.svg" />\n')
    # fitness = \left(\frac{2}{1+e^{K_{on} / -500}} - 1\right) * \left(\frac{2}{1+e^{\Delta F F / -30}}-1\right)
    fid.write(f'<image x="98" y="93" width="44" height="4.4" xlink:href="fitness.svg" />\n')

fid.write('</svg>\n')
fid.write('</div></img></div>\n')

fid.write('</body>\n')
fid.write('''
<script>
var texts = document.querySelectorAll("text");
for (var i = 0; i < texts.length; i++) {
  makeBG(texts[i]);
};\n''')
fid.write(f'var g_nAxes = {max(dataColumns)+1};\n') # yeah, I know.  global variables.
fid.write(f'var g_xaxis = {xaxis};\n')
fid.write(f'var g_yaxis = {yaxis};\n')
fid.write(f'enableXAxis({xaxis});\n')
fid.write(f'enableYAxis({yaxis});\n')
fid.write('</script>\n')
fid.write('</html>')
fid.close()
    
    
