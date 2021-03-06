?	??????i@??????i@!??????i@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??????i@A??ǘ?>@1?&ݖ?Yd@A?C?.lͮ?I*?TU'@*?????LL@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateQ?|a2??!?Y?EIB@)??~j?t??1?????@@:Preprocessing2U
Iterator::Model::ParallelMapV2F%u???!?c?iQR7@)F%u???1?c?iQR7@:Preprocessing2F
Iterator::Model}гY????!`".7yG@)?{??Pk??1?????6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~j?t?x?!vZ?ԏ3%@)?J?4q?1?~?)??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Zd;??!???Ȇ?J@){?G?zd?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}]?!?l?q	@)??H?}]?1?l?q	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I+???!A?2?CoC@)??_?LU?1????_@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor????MbP?!H#ƿD??)????MbP?1H#ƿD??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ???F?!?q????)Ǻ???F?1?q????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???Y??4@Q?????S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A??ǘ?>@A??ǘ?>@!A??ǘ?>@      ??!       "	?&ݖ?Yd@?&ݖ?Yd@!?&ݖ?Yd@*      ??!       2	?C?.lͮ??C?.lͮ?!?C?.lͮ?:	*?TU'@*?TU'@!*?TU'@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???Y??4@y?????S@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropI???????!I???????0"&
CudnnRNNCudnnRNN?ܻ??z??!??J?w???"(
gradients/AddNAddNl???2:??!-*?|`K??"C
$gradients/transpose_9_grad/transpose	Transposep?l4?H??!{?^?????"*
transpose_9	Transpose?3?_?Ɂ?!J%޲????"*
transpose_0	Transposeo??e?|?!?6~???"A
"gradients/transpose_grad/transpose	Transpose?"}??{?!{???L??"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad??B??Em?!޽???i??";
"sequential/dropout_2/dropout/Mul_1Mul?m??e?!Lx??~??"G
.gradient_tape/sequential/dropout_1/dropout/MulMul??9?d?!f5??????Q      Y@Y?h̝?<??a^Έ??X@q?U?˗=W@y728?lRS?"?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 