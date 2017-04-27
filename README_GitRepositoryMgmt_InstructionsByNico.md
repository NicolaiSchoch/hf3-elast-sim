
# ========================= #
# Git-Repository Management #
# ========================= #


By means of the Repository called "hf3-elast-sim"

First, install Git Software:
# sudo apt-get install git-core gitk git-doc


Change into respective folder:
#  cd Workspace/GitHubPlayground/

Clone respective repository on Github into folder:
#  git clone https://github.com/NicolaiSchoch/hf3-elast-sim.git
#  cd hf3-elast-sim/

First steps: Configure new Git-Versioning Structure:
#  git config --help
#  git config --local user.name "NicolaiSchoch"
#  git config --local user.email nicolai.schoch@iwr.uni-heidelberg.de
#  git config --local color.ui true

Use 'branch' and 'status' for navigation:
#  git branch
#  git status
#  git diff
#  git log


Team Hacker Workflow:
=====================

1) Create (-b) a new branch ('new-features') and switch to it by
#  git checkout -b new-features

2) Push this newly created branch on the server (i.e. Github) by
#  git push origin new-features:new-features

Optionally: for another user: Pull this branch by
#  git fetch origin new-features:new-features

3) Create a purely local development branch ('nico-local') from ('new-features'):
#  git checkout -b new-features-nico-local new-features 

------------
4) Hack some files (e.g., add a test.txt file, or add your name to the contributors' list in the README.md file)

5) Commit newly hacked files with to the local branch
#  git add test.txt
#  git commit -m "commit message for explanation"

Possibly repeatedly execute 4) and 5)...

6) Update 'new-features' branch with commited changes by:
a) Update own local version of 'new-features' branch with
#  git checkout new-features
#  git pull origin new-features:new-features

b) Rebase own 'nico-local' branch to the 'new-features' branch with
#  git checkout new-features-nico-local 
#  git rebase new-features

Possibly resolve some merge conflicts at this point...

c) Switch back to the local 'new-features' branch and merge it with own 'nico-local' branch by
#  git checkout new-features
#  git merge new-features-nico-local

d) Now push 'new-features' branch onto the server (i.e. Github) with
#  git push origin new-features:new-features 
So that other developers can pull from it.

7) Switch back to own 'nico-local' branch with
#  git checkout new-features-nico-local
and repeat steps 4) to 7) until done with work.
---------------

8) Now merge local 'new-features' back into 'master' by:
a) Update own local master branch:
#  git checkout master 
#  git pull
b) Merge in own 'new-features' branch by
#  git merge new-features
Possibly resolve some merge conflicts at this point...
c) Push the local master branch to the server (i.e. Github)
#  git push origin master:master 
d) Update the local and server 'new-features' branch
#  git checkout new-features
#  git merge master 
#  git push origin new-features:new-features 

Optionally: for another user: Update the user's local 'new-features' branch by
#  git checkout new-features
#  git pull origin new-features:new-features


#  history > git_documentation.md



