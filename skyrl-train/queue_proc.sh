#!/bin/bash

PID=715757  # Replace with actual PID
echo "Waiting for process $PID to finish..."
while kill -0 $PID 2>/dev/null; do
  sleep 2
  echo "Waiting for process $PID to finish..."
done

echo "Process finished. Running next script..."
bash examples/multiply/run_multiply.sh
