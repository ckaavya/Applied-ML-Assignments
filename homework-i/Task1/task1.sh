#!/bin/sh

mkdir amlagit
cd amlagit

git init

touch 1 2 3 4 5 6 7 8
git add 1
git commit -m 'Adding 1'
git add 2
git commit -m 'Adding 2'
git add 3
git commit -m 'Adding 3'
git add 4
git commit -m 'Adding 4'
git add 5
git commit -m 'Adding 5'

git branch feature HEAD~4
git checkout feature

git add 6
git commit -m 'Adding 6'
git add 7
git commit -m 'Adding 7'
git add 8
git commit -m 'Adding 8'

git checkout master
git branch debug HEAD~2
git rebase --onto feature debug master

git checkout debug
touch 9
git add 9
git commit -m 'Adding 9'

git checkout feature~1 7
git commit --amend -m 'Adding 7 to 9'
