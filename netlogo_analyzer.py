
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
import mpld3
from mpld3 import plugins

# to deal with non-ascii characters
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import unidecode

target_directory='jplag/submissions/.'

commands = ['if','while','set','let','rt|right','lt|left','fd','ca|clear-all','crt|create-turtles','ct|clear-turtles','cp|clear-patches','setxy','reset-ticks','die','hatch','ask','tick','color','nobody','random','pcolor','patch ','count','and','or',' = ', ' != ','of','other','false','true','mod', 'one-of','turtles-own','loop','repeat','red','green','blue','turtle ','patch-here','patch-ahead','heading','who','shape']
block_start = '['
block_stop = ']'
fct_start = 'to|to-report'
fct_stop = 'end'
end_signal = '@#$#@#$#@'

commands = np.array(commands)
cmd_sz0 = commands.size

def strfind(string1, string2):
   if re.search('\|', string1): # commands with 2 possible spellings
     return strfind(string1.split('|')[0], string2) or strfind(string1.split('|')[1], string2)
   if re.search(r"\b" + re.escape(string1) + r"\b", string2):
      return True
   return False

def get_instructions(fname,lines):
  cmds = np.zeros(commands.size)
  found = False
  for l in lines:
    if strfind(fname,l) and strfind(fct_start,l):
      found = True
    
    if strfind(fct_stop,l) and found:
      return cmds
    
    if found:
      idx = [strfind(fct,l) for fct in fct_names]
      # another function call within a function
      # /!\ can be problematic if a function is recursive,
      # so we do not allow for the same name
      if any(idx) and not strfind(fct_start,l): 
        for f in np.array(fct_names)[np.where(idx)]:
          if(f!=fname):
            cmds += get_instructions(f,lines)
          else:
            print('/!\\ recursive call found for function '+fname)
      
      idx = [strfind(cmd,l) for cmd in commands]
      if any(idx):
        cmds[np.where(idx)] += 1
      
  if not found:
    print('WARNING: function',fname,'not found')
  else:
    print('WARNING: function parsing terminated without reaching end of function',fname)
  return cmds
  

files = []
for f in os.listdir(target_directory):
  if f.endswith('.nlogo'):
    files.append(f)

fig0 = plt.figure(figsize=(10,6))
ax0 = fig0.add_subplot(111)
signature = []
for item in files:
  fct_names = []
  lines = []
  with open(os.path.join(target_directory,item),'r') as f:
    print "analyzing file " + item + " ..."
    for line in f:
      line = unidecode.unidecode(unicode(line)) # replace accented letters
      line = line.lower() # only work with lowercase (netlogo is not case sensitive)
      line = line.split(';',1)[0] # remove comments 
      if end_signal in line:
        break
      lines.append(line)
      if strfind(fct_start,line):
        line = re.sub(' +', ' ', line) # remove extra white spaces
        fname = re.findall(r"[\w-]+", line)[1]
        fct_names.append(fname)
  
  # for all function calls, count instructions
  instructions = np.zeros(commands.size)
  not_called = np.array([True]*len(fct_names))
  for l in lines:
    idx = [strfind(fct,l) for fct in fct_names]
    if any(idx) and not strfind(fct_start,l):
      not_called[np.where(idx)] = False
  # get instructions from function that are never called (main and buttons)
  for f in np.array(fct_names)[np.where(not_called)]:
    instructions += get_instructions(f,lines)
  
  instructions = np.array(instructions)
  print(item)
  print(instructions)
  print(commands)
  print('')
  
  # plot some results
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.plot(np.arange(cmd_sz0),instructions,'-o')
  plt.xticks(np.arange(cmd_sz0), commands[:cmd_sz0], rotation='vertical')
  plt.legend(['total calls'])
  plt.grid()
  plt.tight_layout()
  plt.savefig('plots/'+item.split(".nlogo",1)[0]+'.png')
  fig.clf()
  plt.close(fig)

  ax0.plot(np.arange(cmd_sz0),instructions,'-o')
  signature.append(instructions)

files_noext = [f.split(".nlogo")[0] for f in files]

plt.grid()
plt.legend(files_noext)
plt.xticks(np.arange(cmd_sz0), commands, rotation='vertical')
plt.tight_layout()
plt.savefig('plots/total.png')
fig0.clf()
plt.close(fig0)

print "commands summary:"
print(np.sum(np.array(signature),axis=0))

sz = len(signature)
diff = np.zeros((sz,sz))
for s1 in range(sz):
  for s2 in range(sz):
    diff[s1,s2] = np.sqrt( np.sum((signature[s1] - signature[s2])**2) )


ind_sort = np.argsort(diff, axis=None)[::-1]
d_max = np.max(diff)
L = len(files_noext)
for idx in ind_sort:
  if(idx%L == idx/L):
    continue
  print files_noext[idx%L] + " and " + files_noext[idx/L] + " score: " + str((d_max-diff.flatten()[idx])/d_max*100.)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
im = ax.pcolormesh(diff)
for axis in [ax.xaxis, ax.yaxis]:
  axis.set(ticks=np.arange(0.5,sz), ticklabels=files_noext)
plt.xticks(rotation=90)
plt.grid()
plt.colorbar(im)
plt.tight_layout()
plt.savefig('plots/distance_matrix.png')
plt.show()

pca = PCA()
pca.fit(diff)
print(pca.explained_variance_ratio_)
diff_pca = pca.fit_transform(diff)[:,:2]

colors = cm.hsv(np.linspace(0,1,diff_pca.shape[0]))
plt.figure(figsize=(10,8))
for k,c in zip(range(diff_pca.shape[0]), colors):
  plt.scatter(diff_pca[k,0],diff_pca[k,1],color=c)
  plt.annotate(files_noext[k],(diff_pca[k,0],diff_pca[k,1]))
plt.grid()
plt.tight_layout()
plt.savefig('plots/pca_matrix.png')
plt.show()

fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'), figsize=(12,8))
scatter = ax.scatter(diff_pca[:,0],diff_pca[:,1], color=colors, s=120)
ax.grid(color='white', linestyle='solid')
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=files_noext)
mpld3.plugins.connect(fig, tooltip)
mpld3.save_html(fig,'plots/pca_matrix.html')
mpld3.show()





