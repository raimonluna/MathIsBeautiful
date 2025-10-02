import os
import re
import numpy as np
from glob import glob

# from deep_translator import GoogleTranslator
# translator = GoogleTranslator(source='auto', target='es')

# for i in *.png; do convert $i -resize 512x ../LowQuality/$i; done;

######################### RETRIEVE INFORMATION #########################

images_dirs   = np.sort(glob('../Output/LowQuality/MIB*.png'))
captions_dirs = np.sort(glob('../MIB*/**/*.txt', recursive = True))

captions = []
for caption_file in captions_dirs:
    with open(caption_file) as f:
        captions += f.read().split(65 * '=')
captions = [c.strip('\n') for c in captions]
captions = [c for c in captions if len(c) > 0]
captions = [c.replace("https", "\\url{https") for c in captions]
captions = [c.replace(".py", ".py}") for c in captions]
captions = [c.replace("\n\n", "\\\\\n\n") for c in captions]

############################ WRITE THE FILE ############################

header = r""" 
\documentclass[12pt,landscape]{article}

% Geometry and layout
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{graphicx}
\usepackage{parskip} % nicer paragraph spacing
\usepackage{multicol}
\usepackage{caption}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage{hyperref}
\usepackage{amsfonts}

% Font & style
\usepackage{palatino} % elegant font
\renewcommand{\baselinestretch}{1.2}

% TOC formatting (minimalistic look)
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecfont}{\large}
\renewcommand{\cftsecpagefont}{\large}

% Section formatting
\titleformat{\section}{\LARGE\bfseries}{\thesection}{1em}{}

% --- PAGE TEMPLATE ---
\newcommand{\artpage}[3][]{%
  \begin{minipage}[t]{0.48\linewidth}
    \vspace{0pt} % align top
    \section*{#2} % artwork title
    \addcontentsline{toc}{section}{#2} % add to TOC
    #3 % text content
  \end{minipage}%
  \hfill
  \begin{minipage}[t]{0.48\linewidth}
    \vspace{0pt} % align top
    \centering
    \includegraphics[width=\linewidth,height=\linewidth,keepaspectratio]{#1}
  \end{minipage}%
  \newpage
}

% --- DOCUMENT START ---
\begin{document}

% --- COVER PAGE ---
\begin{titlepage}
    \centering
    \vspace*{3cm}
    {\Huge\bfseries Mathematical Art Collection \par}
    \vspace{1cm}
    {\LARGE A Journey Through Patterns and Geometry\par}
    \vspace{2cm}
    {\Large Raimon Luna\par}
    \vfill
    {\Large \today\par}
\end{titlepage}

% --- TABLE OF CONTENTS ---
\thispagestyle{empty}
\begin{center}
    \vspace*{1cm}
    {\Huge \bfseries Contents \par}
\end{center}
\vspace{1cm}
\tableofcontents
\newpage

"""

os.remove("demofile.tex")
with open("demofile.tex", "a") as f:
  f.write(header)
  for caption, image_dir in zip(captions, images_dirs):
      title = re.sub(r"(\w)([A-Z])", r"\1 \2", image_dir.split('/')[-1][8:-4])
      artpage = "\n\\artpage[" + image_dir + "]{" + title + "}{%\n" + caption + "\n}\n"
      f.write(artpage)
  f.write("\n\\end{document}")

