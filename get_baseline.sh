pipenv shell

experiment="61frags-med"
model="mfcconformer"
fs="2000"
timestamp=$(date +"%b%d-%Y")
seconds_since_midnight=$(($(date +%s) % 86400))
today="${timestamp}-${seconds_since_midnight}"
name="vest-data-${model}-baseline-${today}"
channels="1,2,3,4"
log_file="${name}-123-ch${channels//,/}-${experiment}.log"
schedule_file="data/th123-schedule-from-spec-matt-med-1234.json"
num_channels=$(( $(grep -o "," <<< "$channels" | wc -l) + 1))

echo "" > "${log_file}"
echo "Schedule file:" >> ${log_file}
cat ${schedule_file} >> ${log_file}
echo "" >> ${log_file} 

for idx in {1..10}; do
  echo "Trial ${idx}" >> "${log_file}"
  python src/python/heartsignals/run_model.py train-audio-model \
    -L 4 \
    -G ${fs} \
    -A time \
    -B vest-data-matt \
    -S ${schedule_file} \
    -O models/${name}-${idx} \
    -I data/preprocessed_audio/vest-data-baseline-4-123-30f/ \
    -M ${model} \
    -H ${channels} \
    --folds 5  \
    >> "${log_file}"
done
