git diff --quiet coverage.svg
if [ $? -eq 1 ] 
then
    echo difference in badge... adding
    git add coverage.svg
    git commit -m"updating coverage badge"
    git push
else
    echo no difference in badge... ignoring
fi