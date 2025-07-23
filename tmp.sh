
while ps -p 44602 > /dev/null; do
  sleep 10
done
python3 exp_orca.py