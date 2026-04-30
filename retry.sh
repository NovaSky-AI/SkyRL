SCRIPT_PATH=$1 
LOG_FILE=$2 
state=error
error_string="unexpected system error"
echo "Clearing log file first $LOG_FILE"
> $LOG_FILE
while [ $state == 'error' ]; do 
    echo "(Re)Running the script $SCRIPT_PATH"
    bash $SCRIPT_PATH >> $LOG_FILE 2>&1
    # script returned, let's check if there was an error
    last_lines=$(tail -n 100 $LOG_FILE)
    if [[ ! "$last_lines" =~ "$error_string" ]];then
        state=noterror
    fi
done