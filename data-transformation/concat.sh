set -e

readarray -t FILES <<< `rsync --list-only --exclude="@*" --exclude=".DS_Store" --exclude="./" $RSYNC_REMOTE/$1/ | awk '{ $1=$2=$3=$4=""; print substr($0,5); }'`

if [ -n "$2" ]; then
    first_file=$(printf "%s\n" "${FILES[@]}" | grep --line-number --fixed-strings -- "$2.mp4" | cut -d: -f1)-1;
else
    first_file="1"
fi

if [ -n "$3" ]; then
    last_file=$(printf "%s\n" "${FILES[@]}" | grep --line-number --fixed-strings -- "$3.mp4" | cut -d: -f1);
else
    last_file="1"
fi

slice_length=$((last_file - first_file))
new_name=${FILES[$first_file]%.*}-${FILES[$last_file-1]%.*}.mp4
echo "⏳ Joining following range of files ($slice_length) into '$new_name':"
for i in ${FILES[@]:$first_file:$slice_length}; do echo "  $i"; done

mkdir -p old
for i in ${FILES[@]:$first_file:$slice_length}; do
    rsync -Pr --info=progress2,flist0,name0,stats0 $RSYNC_REMOTE/$1/$i old/$i
done

for i in ${FILES[@]:$first_file:$slice_length}; do printf "file '%s'\n" "$PWD/old/$i"; done | ffmpeg -f concat -y -safe 0 -protocol_whitelist file,pipe -i /dev/stdin -c copy old/$new_name
for i in ${FILES[@]:$first_file:$slice_length}; do rm -rf old/$i; done

ffmpeg -v error -stats -y -i "old/$new_name" $new_name
rm -rf old/$new_name

rsync -Pr --info=progress2,flist0,name0,stats0 $new_name $RSYNC_REMOTE/$1/
for i in ${FILES[@]:$first_file:$slice_length}; do rsync -r --info=progress2,flist0,name0,stats0 --delete --include=$i '--exclude=*' . $RSYNC_REMOTE/$1/; done

echo "✅ Successfully joined files into '$new_name'"

rm -rf $new_name
rmdir old
