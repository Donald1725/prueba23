from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Inicializa una sesión de Spark
spark = SparkSession.builder.appName("FuzzyCMeans").getOrCreate()

# Inicializa la matriz de pertenencia
def inicializar_matriz_pertenencia(n_muestras, n_clusters):
    return spark.createDataFrame([(Vectors.dense([0.0] * n_clusters),) for _ in range(n_muestras)], ["matriz_pertenencia"])

# Actualiza la matriz de pertenencia
def actualizar_matriz_pertenencia(datos, centroides, difusidad):
    matriz_distancia = datos.crossJoin(centroides) \
        .select("features", "centroides") \
        .rdd \
        .map(lambda row: (row.features, row.centroides, float(F.norm(row.features - row.centroides)))) \
        .toDF(["features", "centroides", "distancia"]) \
        .groupBy("features") \
        .agg(F.collect_list("distancia").alias("distancias")) \
        .withColumn("distancias", F.array([1 / (d ** (2 / (difusidad - 1))) for d in F.col("distancias")])) \
        .withColumn("suma_distancias", F.expr("aggregate(distancias, 0D, (acc, x) -> acc + x)")) \
        .withColumn("matriz_pertenencia", F.expr("transform(distancias, x -> x / suma_distancias)")) \
        .select("features", "matriz_pertenencia")

    return matriz_distancia

# Actualiza los centroides
def actualizar_centroides(datos, matriz_pertenencia, difusidad):
    centroides = datos.withColumn("matriz_pertenencia", F.expr(f"transform(matriz_pertenencia, x -> pow(x, {difusidad}))")) \
        .groupBy().agg(F.sum("matriz_pertenencia").alias("suma_matriz_pertenencia"),
                       F.sum(F.expr("matriz_pertenencia * features")).alias("suma_pesada")) \
        .select(F.expr("suma_pesada / suma_matriz_pertenencia").alias("centroides"))

    return centroides

# Implementa el algoritmo Fuzzy C-Means
def fuzzy_c_means(datos, n_clusters, difusidad, max_iters=100, tolerancia=1e-4):
    n_muestras, n_caracteristicas = datos.select("features").first()[0].size
    matriz_pertenencia = inicializar_matriz_pertenencia(n_muestras, n_clusters)
    centroides = datos.limit(n_clusters).select("features").rdd.map(lambda row: row.features).toDF("centroides")

    for _ in range(max_iters):
        centroides_antiguos = centroides.copy()

        matriz_pertenencia = actualizar_matriz_pertenencia(datos, centroides, difusidad)
        centroides = actualizar_centroides(datos, matriz_pertenencia, difusidad)

        norm_diff = centroides_antiguos.crossJoin(centroides) \
            .select(F.expr("sum(abs(centroides1 - centroides2)) as diff")) \
            .first()[0]

        if norm_diff < tolerancia:
            break

    return centroides, matriz_pertenencia

# Carga los datos desde un archivo CSV
df = spark.read.csv("hdfs://master:9000/user/hadoop/contaminantes/datafinal.csv", sep=';', header=True, inferSchema=True)

# Convierte las columnas a numéricas, reemplazando las comas por puntos y convirtiendo los errores en NaN
for col_name in ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']:
    df = df.withColumn(col_name, F.col(col_name).cast("double"))

# Crea un DataFrame con la columna "features"
feature_cols = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df).select("features")

n_clusters = 3
difusidad = 2

df_final = df_assembled

# Agrega columnas de grupos al DataFrame final
for col_name in ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']:
    datos_col = df.select(col_name).withColumnRenamed(col_name, "features")
    _, matriz_pertenencia_col = fuzzy_c_means(datos_col, n_clusters, difusidad)
    col_grupo = matriz_pertenencia_col.select(F.expr("argmax(matriz_pertenencia) as grupo"), "features")
    df_final = df_final.join(col_grupo, "features")

# Muestra los resultados
df_final.show(truncate=False)

# Guardar el DataFrame 'df_spark' en un archivo de csv
df_final.write.format('com.databricks.spark.csv').option('header', 'true').save('hdfs://master:9000/user/hadoop/contaminantes/clusters_resultado.csv')

# Detén la sesión de Spark
spark.stop()