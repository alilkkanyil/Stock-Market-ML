?	n/?m@n/?m@!n/?m@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-n/?m@?"R??=@1\?M??g@A?1?????I`???Y&@*	     @O@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenatetF??_??!ףp=
C@)w-!?l??1??Q??A@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!??????9@)????Mb??1??????9@:Preprocessing2F
Iterator::ModelS?!?uq??!q=
ףpE@)??_vO??1H?z?G1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q?~?!      (@)?g??s?u?1?(\??? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4?8EG??!???(\?L@)?~j?t?h?1333333@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!)\???(@)/n??b?1)\???(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap ?o_Ι?!)\???(D@)Ǻ???V?1?Q???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice????MbP?!????????)????MbP?1????????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor??H?}M?!
ףp=
??)??H?}M?1
ףp=
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?Y??1@Q9?i8Y?T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?"R??=@?"R??=@!?"R??=@      ??!       "	\?M??g@\?M??g@!\?M??g@*      ??!       2	?1??????1?????!?1?????:	`???Y&@`???Y&@!`???Y&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Y??1@y9?i8Y?T@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??T/???!??T/???0"&
CudnnRNNCudnnRNN;??J(???!?jyCO??"(
gradients/AddNAddN?^????!yI ?7???"C
$gradients/transpose_9_grad/transpose	TransposeA??????!???????"*
transpose_9	Transpose?e]???~?!?ט]?'??"*
transpose_0	Transpose̚*?-?x?!?,]??X??"A
"gradients/transpose_grad/transpose	TransposeH??aw?!F?{????"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradBe?*??j?!fʸl{???"G
.gradient_tape/sequential/dropout_2/dropout/MulMul[p??8a?!?:N?????";
"sequential/dropout_1/dropout/Mul_1Mulż?a4a?!=HC????Q      Y@Y?h̝?<??a^Έ??X@q?)o?RW@y??]V?2P?"?
both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 