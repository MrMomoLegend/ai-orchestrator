import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ACC='#1F3864'; MODEL='#2E6F4E'; STORE='#8A5A00'; LIGHT='#F4F6FA'

fig, ax = plt.subplots(figsize=(11,6.6))
ax.set_xlim(0,110); ax.set_ylim(0,68); ax.axis('off')

# local boundary
ax.add_patch(Rectangle((3,2),104,58, fill=False, ls=(0,(6,4)), lw=2.0, ec='#B03030'))
ax.text(5.5,60.6,'LOCAL MACHINE: no external API calls', ha='left', va='bottom',
        fontsize=10.5, color='#B03030', fontweight='bold')

def box(x,y,w,h,title,sub='',fc='white',ec=ACC,tc=ACC,fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle='round,pad=0.6,rounding_size=1.2',
                                fc=fc, ec=ec, lw=1.6))
    if sub:
        ax.text(x+w/2, y+h/2+1.5, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
        ax.text(x+w/2, y+h/2-2.2, sub, ha='center', va='center', fontsize=8.4, color='#555555')
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)

def arrow(x1,y1,x2,y2,label='',rad=0.0,color='#666666',off=(0,1.6),fs=8.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>', mutation_scale=13,
                                 lw=1.5, color=color, connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((x1+x2)/2+off[0], (y1+y2)/2+off[1], label, ha='center', va='center',
                fontsize=fs, color='#444444', style='italic')

# frontend
box(6,34,22,18,'React frontend','Vite · single page', fc=LIGHT)
ax.text(17,31.0,'FR1 text  ·  FR2 voice  ·  FR3 upload', ha='center', fontsize=7.8, color='#555')
ax.text(17,28.2,'FR5 answer  ·  FR6 sources', ha='center', fontsize=7.8, color='#555')

# orchestrator
box(40,30,26,26,'FastAPI\norchestrator','answer_question()')
ax.text(53,33.4,'/ask  /transcribe  /ask/audio', ha='center', fontsize=7.6, color='#555')
ax.text(53,31.2,'/upload  /documents  /health', ha='center', fontsize=7.6, color='#555')

# models
box(78,46,26,11,'Whisper','faster-whisper base.en · int8', ec=MODEL, tc=MODEL, fs=10)
box(78,30,26,11,'ChromaDB','all-MiniLM-L6-v2 · cosine', ec=STORE, tc=STORE, fs=10)
box(78,14,26,11,'Llama 3.1 (8B)','via Ollama', ec=MODEL, tc=MODEL, fs=10)

# corpus
box(40,8,26,12,'Document corpus','28 files → 3,215 chunks', ec=STORE, tc=STORE, fs=10)
ax.text(53,10.4,'embedded into ChromaDB', ha='center', fontsize=7.6, color='#8A5A00', style='italic')

arrow(28.6,46,39.4,49,'voice / text / upload',rad=0.05,off=(0,2.0))
arrow(39.4,37,28.6,40,'answer + sources',rad=0.05,off=(0,-2.4))
arrow(66.6,50,77.4,51,'audio',rad=0.05,off=(0,1.6))
arrow(77.4,48,66.6,47,'transcript',rad=0.05,off=(0,-2.0))
arrow(66.6,38,77.4,37,'query embedding',rad=0.05,off=(0,1.6))
arrow(77.4,34,66.6,33.6,'top-k chunks + distance',rad=0.05,off=(0,-1.8))
arrow(53,20.6,53,29.4,'ingest',rad=0.0,off=(4.0,0))
arrow(66.6,31,77.4,23,'grounded prompt',rad=0.0,off=(2.0,-2.0))
arrow(77.4,19,66.6,30.5,'answer',rad=0.0,off=(-3.0,-2.2))

# threshold callout
ax.add_patch(FancyBboxPatch((6,8),28,14, boxstyle='round,pad=0.6,rounding_size=1.2',
                            fc='#FFF4E0', ec='#E0A030', lw=1.5))
ax.text(20,18.6,'Retrieval-confidence threshold', ha='center', fontsize=9.2, fontweight='bold', color='#7A4E00')
ax.text(20,14.6,'if nearest cosine distance > 0.53:\nrefuse before the model is called', ha='center',
        fontsize=8.2, color='#7A4E00', linespacing=1.5)
ax.add_patch(FancyArrowPatch((34.6,17.5),(42,31.5), arrowstyle='-|>', mutation_scale=11,
                             lw=1.3, color='#E0A030', ls=(0,(4,2)), connectionstyle='arc3,rad=-0.2'))

ax.legend(handles=[Line2D([],[],color=MODEL,lw=3,label='Pre-trained model'),
                   Line2D([],[],color=STORE,lw=3,label='Storage / retrieval'),
                   Line2D([],[],color=ACC,lw=3,label='Application code')],
          loc='lower right', bbox_to_anchor=(1.0,-0.02), frameon=False, fontsize=8.6, ncol=3)
plt.tight_layout()
plt.savefig('figures/fig1_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print('arch ok')

# ---------------- Gantt ----------------
fig, ax = plt.subplots(figsize=(11,5.4))
tasks=[
 ('Design; FastAPI skeleton',6,8,6,8),
 ('Llama 3.1 + RAG prototype; PPR (W10)',9,10,9,10),
 ('Whisper integration; end-to-end voice',11,12,17,17),
 ('React frontend; integration polish',13,15,17,17),
 ('Evaluation harness + experiments 5.2-5.6',16,17,17,18),
 ('Draft report',17,18,17,18),
 ('Experiments 5.7-5.10; pytest suite',16,17,18,19),
 ('Improvements; exam prep',19,22,19,22),
 ('Final report, code, demo video',23,24,23,24),
]
for i,(name,ps,pe,as_,ae) in enumerate(tasks):
    y=len(tasks)-i
    ax.barh(y+0.18, pe-ps+1, left=ps, height=0.32, color='#C9D2E4', edgecolor='#9AA9C6', label='Planned' if i==0 else '')
    late = as_>ps or ae>pe
    ax.barh(y-0.18, ae-as_+1, left=as_, height=0.32,
            color=('#C58A2E' if late else ACC), edgecolor='none',
            label=('Actual' if i==0 else ''))
ax.set_yticks([len(tasks)-i for i in range(len(tasks))])
ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
ax.set_xlabel('Project week', fontsize=10)
ax.set_xlim(5.4,25.2); ax.set_xticks(range(6,25))
ax.grid(axis='x', color='#E6E6E6', lw=0.8); ax.set_axisbelow(True)
for s in ('top','right','left'): ax.spines[s].set_visible(False)
for wk,lab in [(10,'PPR'),(18,'Draft'),(21,'Exam'),(24,'Final')]:
    ax.axvline(wk, color='#E0A030', ls=(0,(5,3)), lw=1.3)
    ax.text(wk, len(tasks)+0.85, lab, ha='center', fontsize=8.4, color='#7A4E00', fontweight='bold')
h=[plt.Rectangle((0,0),1,1,color='#C9D2E4'), plt.Rectangle((0,0),1,1,color=ACC), plt.Rectangle((0,0),1,1,color='#C58A2E')]
ax.legend(h,['Planned','Actual (on or ahead of plan)','Actual (slipped)'], loc='lower left',
          bbox_to_anchor=(0,-0.30), ncol=3, frameon=False, fontsize=8.8)
ax.set_title('Project schedule: planned against actual (weeks 6-24)', fontsize=11.5, color=ACC, fontweight='bold', pad=22)
plt.tight_layout()
plt.savefig('figures/fig2_gantt.png', dpi=300, bbox_inches='tight', facecolor='white')
print('gantt ok')
