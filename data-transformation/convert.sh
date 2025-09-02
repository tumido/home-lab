#!/bin/sh
set -e

readarray -t FILES <<< `rsync --list-only --exclude="@*" --exclude=".DS_Store" --exclude="./" $RSYNC_REMOTE/$1/ | awk '{ $1=$2=$3=$4=""; print substr($0,5); }'`
mkdir -p old

if [ -n "$2" ]; then
    start_from=$(printf "%s\n" "${FILES[@]}" | grep --line-number --fixed-strings -- "$2" | cut -d: -f1)-1;
    echo "⏳ Starting from: '${FILES[$start_from]}'"
else
    start_from="1"
    echo "⏳ Starting from the first file in a directory: '${FILES[$start_from]}'"
fi


prefix_emoji () { awk -v RS='\r' "{ printf \"$1 %s\r\", \$0; fflush() } END{ print \"\" }"; }

for i in ${FILES[@]:$start_from}; do
    echo "⏳ Processing: '$i'"
    if [[ $i != *.* ]]; then
        echo "⏭️ Skipping $i"
        continue
    fi
    new_name=${i%.*}.mp4
    rsync -Pr --info=progress2,flist0,name0,stats0 $RSYNC_REMOTE/$1/$i old/$i
    ffmpeg -v error -stats -y -i "old/$i" "$new_name" 2> >(awk -v RS='\r' '{ printf "🎬 %s\r", $0; fflush() } END{ print "" }')
    rsync -Pr --info=progress2,flist0,name0,stats0 $new_name $RSYNC_REMOTE/$1/
    echo "✅ Successfully converted '$i' to '$new_name'"
    rm -rf old/$i $new_name
done
rmdir old
