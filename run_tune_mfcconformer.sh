pipenv shell

model="mfcconformer"
fs="2000"
today=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')

python src/python/heartsignals/run_model.py tune-mfcconformer \
  -L 4 \
  -G ${fs} \
  -A time \
  -B vest-data-matt \
  -S data/th123-schedule-from-spec-matt-med-1234.json \
  -I data/preprocessed_audio/vest-data-baseline-4-123-30f/ \
  -M ${model} \
  --db_password "MyDatabasePass"