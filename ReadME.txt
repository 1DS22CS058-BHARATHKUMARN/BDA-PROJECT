terminal1:$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties
terminal2:$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
terminal3:$KAFKA_HOME/bin/kafka-topics.sh --create --topic HeartData --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
	cmd 2:$KAFKA_HOME/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
    hdfs dfs -mkdir /heart_data
    hdfs dfs -mkdir /heart_predictions_stream  //folders to store data
    cmd 3:start-dfs.sh
connecting local system to ubuntu terminal
cd /mnt/c/Users/bhara/OneDrive/Desktop/heart-disease-pipeline
source heart-env/bin/activate
terminal4:python producer.py
terminal5:spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.spark:spark-token-provider-kafka-0-10_2.12:3.5.1 \
  consumer_spark.py

  hdfs dfs -ls /heart_data/
hdfs dfs -cat /heart_data/part-00000*.csv | head  //checking HeartData
terminal6:spark-submit train_model.py
    ls -lh heart_model.pkl  //checking

    spark-submit   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.spark:spark-token-provider-kafka-0-10_2.12:3.5.1   consumer_predict.py
    //predictions

    in zip file--> ignore hear_output,predictions_local

