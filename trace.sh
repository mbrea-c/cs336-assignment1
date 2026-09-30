time -v -o summary.txt \
    psrecord "$*" \
    --include-children \
    --interval 1 --log usage.txt \
    --plot usage.png
