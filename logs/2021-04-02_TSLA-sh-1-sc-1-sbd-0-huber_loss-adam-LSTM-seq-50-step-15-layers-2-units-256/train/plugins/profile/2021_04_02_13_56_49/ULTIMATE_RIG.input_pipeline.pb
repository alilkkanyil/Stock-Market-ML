	 |(ђ?+@ |(ђ?+@! |(ђ?+@	f??_@f??_@!f??_@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6 |(ђ?+@T?J?ó??16<?R??@A?H0??Z??I??2?P\@Y/ޏ?/???*	43333?M@2U
Iterator::Model::ParallelMapV2???߾??!ؖ??`7@)???߾??1ؖ??`7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S㥋?!?袋.?6@)?+e?X??1?ļ?!13@:Preprocessing2F
Iterator::Modela??+e??!+??D@)A??ǘ???1[?R?֯2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2U0*???!?c?xTn:@)'???????1 cţr2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!?Ω?? @){?G?zt?1?Ω?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe?X???!???j?M@)??H?}m?1>???>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!?!1ogH@)?J?4a?1?!1ogH@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?????!?(iv=@)??H?}]?1>???>@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g??_@I??:???G@Qt??"?H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?J?ó??T?J?ó??!T?J?ó??      ??!       "	6<?R??@6<?R??@!6<?R??@*      ??!       2	?H0??Z???H0??Z??!?H0??Z??:	??2?P\@??2?P\@!??2?P\@B      ??!       J	/ޏ?/???/ޏ?/???!/ޏ?/???R      ??!       Z	/ޏ?/???/ޏ?/???!/ޏ?/???b      ??!       JGPUYg??_@b q??:???G@yt??"?H@