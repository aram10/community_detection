??

??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??	
|
dense_144/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ *!
shared_namedense_144/kernel
u
$dense_144/kernel/Read/ReadVariableOpReadVariableOpdense_144/kernel*
_output_shapes

:^ *
dtype0
t
dense_144/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_144/bias
m
"dense_144/bias/Read/ReadVariableOpReadVariableOpdense_144/bias*
_output_shapes
: *
dtype0
|
dense_145/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^*!
shared_namedense_145/kernel
u
$dense_145/kernel/Read/ReadVariableOpReadVariableOpdense_145/kernel*
_output_shapes

: ^*
dtype0
t
dense_145/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_145/bias
m
"dense_145/bias/Read/ReadVariableOpReadVariableOpdense_145/bias*
_output_shapes
:^*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
?
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1

0
1
 
?
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
h

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api

0
1

0
1
 
?
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
VT
VARIABLE_VALUEdense_144/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_144/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_145/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_145/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1

0
1
 
?
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
 
 
 

	0
 

0
1

0
1
 
?
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????^*
dtype0*
shape:?????????^
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_144/kerneldense_144/biasdense_145/kerneldense_145/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_16666289
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_144/kernel/Read/ReadVariableOp"dense_144/bias/Read/ReadVariableOp$dense_145/kernel/Read/ReadVariableOp"dense_145/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_16666795
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_144/kerneldense_144/biasdense_145/kerneldense_145/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_16666817??	
?
?
K__inference_dense_144_layer_call_and_return_all_conditional_losses_16666689

inputs
unknown:^ 
	unknown_0: 
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_144_activity_regularizer_166657882
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_72_layer_call_fn_16666317
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_166661802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
G__inference_dense_145_layer_call_and_return_conditional_losses_16666732

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666046

inputs$
dense_145_16666034: ^ 
dense_145_16666036:^
identity??!dense_145/StatefulPartitionedCall?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
!dense_145/StatefulPartitionedCallStatefulPartitionedCallinputsdense_145_16666034dense_145_16666036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_145_layer_call_and_return_conditional_losses_166659902#
!dense_145/StatefulPartitionedCall?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_145_16666034*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity*dense_145/StatefulPartitionedCall:output:0"^dense_145/StatefulPartitionedCall3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_145/StatefulPartitionedCall!dense_145/StatefulPartitionedCall2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?#
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665966
input_73$
dense_144_16665945:^  
dense_144_16665947: 
identity

identity_1??!dense_144/StatefulPartitionedCall?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
!dense_144/StatefulPartitionedCallStatefulPartitionedCallinput_73dense_144_16665945dense_144_16665947*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122#
!dense_144/StatefulPartitionedCall?
-dense_144/ActivityRegularizer/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_144_activity_regularizer_166657882/
-dense_144/ActivityRegularizer/PartitionedCall?
#dense_144/ActivityRegularizer/ShapeShape*dense_144/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
%dense_144/ActivityRegularizer/truedivRealDiv6dense_144/ActivityRegularizer/PartitionedCall:output:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_144_16665945*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentity*dense_144/StatefulPartitionedCall:output:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_144/ActivityRegularizer/truediv:z:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_73
?
?
G__inference_dense_144_layer_call_and_return_conditional_losses_16665812

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
!__inference__traced_save_16666795
file_prefix/
+savev2_dense_144_kernel_read_readvariableop-
)savev2_dense_144_bias_read_readvariableop/
+savev2_dense_145_kernel_read_readvariableop-
)savev2_dense_145_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_144_kernel_read_readvariableop)savev2_dense_144_bias_read_readvariableop+savev2_dense_145_kernel_read_readvariableop)savev2_dense_145_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: :^ : : ^:^: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:^ : 

_output_shapes
: :$ 

_output_shapes

: ^: 

_output_shapes
:^:

_output_shapes
: 
?B
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666553

inputs:
(dense_144_matmul_readvariableop_resource:^ 7
)dense_144_biasadd_readvariableop_resource: 
identity

identity_1?? dense_144/BiasAdd/ReadVariableOp?dense_144/MatMul/ReadVariableOp?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
dense_144/MatMul/ReadVariableOpReadVariableOp(dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_144/MatMul/ReadVariableOp?
dense_144/MatMulMatMulinputs'dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_144/MatMul?
 dense_144/BiasAdd/ReadVariableOpReadVariableOp)dense_144_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_144/BiasAdd/ReadVariableOp?
dense_144/BiasAddBiasAdddense_144/MatMul:product:0(dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_144/BiasAdd
dense_144/SigmoidSigmoiddense_144/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_144/Sigmoid?
4dense_144/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_144/ActivityRegularizer/Mean/reduction_indices?
"dense_144/ActivityRegularizer/MeanMeandense_144/Sigmoid:y:0=dense_144/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Mean?
'dense_144/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_144/ActivityRegularizer/Maximum/y?
%dense_144/ActivityRegularizer/MaximumMaximum+dense_144/ActivityRegularizer/Mean:output:00dense_144/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/Maximum?
'dense_144/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_144/ActivityRegularizer/truediv/x?
%dense_144/ActivityRegularizer/truedivRealDiv0dense_144/ActivityRegularizer/truediv/x:output:0)dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
!dense_144/ActivityRegularizer/LogLog)dense_144/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/Log?
#dense_144/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_144/ActivityRegularizer/mul/x?
!dense_144/ActivityRegularizer/mulMul,dense_144/ActivityRegularizer/mul/x:output:0%dense_144/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/mul?
#dense_144/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_144/ActivityRegularizer/sub/x?
!dense_144/ActivityRegularizer/subSub,dense_144/ActivityRegularizer/sub/x:output:0)dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/sub?
)dense_144/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_144/ActivityRegularizer/truediv_1/x?
'dense_144/ActivityRegularizer/truediv_1RealDiv2dense_144/ActivityRegularizer/truediv_1/x:output:0%dense_144/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_144/ActivityRegularizer/truediv_1?
#dense_144/ActivityRegularizer/Log_1Log+dense_144/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/Log_1?
%dense_144/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_144/ActivityRegularizer/mul_1/x?
#dense_144/ActivityRegularizer/mul_1Mul.dense_144/ActivityRegularizer/mul_1/x:output:0'dense_144/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/mul_1?
!dense_144/ActivityRegularizer/addAddV2%dense_144/ActivityRegularizer/mul:z:0'dense_144/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/add?
#dense_144/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_144/ActivityRegularizer/Const?
!dense_144/ActivityRegularizer/SumSum%dense_144/ActivityRegularizer/add:z:0,dense_144/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/Sum?
%dense_144/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_144/ActivityRegularizer/mul_2/x?
#dense_144/ActivityRegularizer/mul_2Mul.dense_144/ActivityRegularizer/mul_2/x:output:0*dense_144/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/mul_2?
#dense_144/ActivityRegularizer/ShapeShapedense_144/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
'dense_144/ActivityRegularizer/truediv_2RealDiv'dense_144/ActivityRegularizer/mul_2:z:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_144/ActivityRegularizer/truediv_2?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentitydense_144/Sigmoid:y:0!^dense_144/BiasAdd/ReadVariableOp ^dense_144/MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_144/ActivityRegularizer/truediv_2:z:0!^dense_144/BiasAdd/ReadVariableOp ^dense_144/MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_144/BiasAdd/ReadVariableOp dense_144/BiasAdd/ReadVariableOp2B
dense_144/MatMul/ReadVariableOpdense_144/MatMul/ReadVariableOp2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666234
input_1)
sequential_144_16666209:^ %
sequential_144_16666211: )
sequential_145_16666215: ^%
sequential_145_16666217:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?&sequential_144/StatefulPartitionedCall?&sequential_145/StatefulPartitionedCall?
&sequential_144/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_144_16666209sequential_144_16666211*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166658342(
&sequential_144/StatefulPartitionedCall?
&sequential_145/StatefulPartitionedCallStatefulPartitionedCall/sequential_144/StatefulPartitionedCall:output:0sequential_145_16666215sequential_145_16666217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660032(
&sequential_145/StatefulPartitionedCall?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_144_16666209*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_145_16666215*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity/sequential_145/StatefulPartitionedCall:output:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_144/StatefulPartitionedCall:output:13^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_144/StatefulPartitionedCall&sequential_144/StatefulPartitionedCall2P
&sequential_145/StatefulPartitionedCall&sequential_145/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
G__inference_dense_144_layer_call_and_return_conditional_losses_16666760

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
$__inference__traced_restore_16666817
file_prefix3
!assignvariableop_dense_144_kernel:^ /
!assignvariableop_1_dense_144_bias: 5
#assignvariableop_2_dense_145_kernel: ^/
!assignvariableop_3_dense_145_bias:^

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_144_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_144_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_145_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_145_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_dense_145_layer_call_and_return_conditional_losses_16665990

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666646
dense_145_input:
(dense_145_matmul_readvariableop_resource: ^7
)dense_145_biasadd_readvariableop_resource:^
identity?? dense_145/BiasAdd/ReadVariableOp?dense_145/MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
dense_145/MatMul/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_145/MatMul/ReadVariableOp?
dense_145/MatMulMatMuldense_145_input'dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/MatMul?
 dense_145/BiasAdd/ReadVariableOpReadVariableOp)dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_145/BiasAdd/ReadVariableOp?
dense_145/BiasAddBiasAdddense_145/MatMul:product:0(dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/BiasAdd
dense_145/SigmoidSigmoiddense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_145/Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitydense_145/Sigmoid:y:0!^dense_145/BiasAdd/ReadVariableOp ^dense_145/MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_145/BiasAdd/ReadVariableOp dense_145/BiasAdd/ReadVariableOp2B
dense_145/MatMul/ReadVariableOpdense_145/MatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_145_input
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666629

inputs:
(dense_145_matmul_readvariableop_resource: ^7
)dense_145_biasadd_readvariableop_resource:^
identity?? dense_145/BiasAdd/ReadVariableOp?dense_145/MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
dense_145/MatMul/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_145/MatMul/ReadVariableOp?
dense_145/MatMulMatMulinputs'dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/MatMul?
 dense_145/BiasAdd/ReadVariableOpReadVariableOp)dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_145/BiasAdd/ReadVariableOp?
dense_145/BiasAddBiasAdddense_145/MatMul:product:0(dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/BiasAdd
dense_145/SigmoidSigmoiddense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_145/Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitydense_145/Sigmoid:y:0!^dense_145/BiasAdd/ReadVariableOp ^dense_145/MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_145/BiasAdd/ReadVariableOp dense_145/BiasAdd/ReadVariableOp2B
dense_145/MatMul/ReadVariableOpdense_145/MatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_72_layer_call_fn_16666206
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_166661802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?h
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666376
xI
7sequential_144_dense_144_matmul_readvariableop_resource:^ F
8sequential_144_dense_144_biasadd_readvariableop_resource: I
7sequential_145_dense_145_matmul_readvariableop_resource: ^F
8sequential_145_dense_145_biasadd_readvariableop_resource:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?/sequential_144/dense_144/BiasAdd/ReadVariableOp?.sequential_144/dense_144/MatMul/ReadVariableOp?/sequential_145/dense_145/BiasAdd/ReadVariableOp?.sequential_145/dense_145/MatMul/ReadVariableOp?
.sequential_144/dense_144/MatMul/ReadVariableOpReadVariableOp7sequential_144_dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_144/dense_144/MatMul/ReadVariableOp?
sequential_144/dense_144/MatMulMatMulx6sequential_144/dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_144/dense_144/MatMul?
/sequential_144/dense_144/BiasAdd/ReadVariableOpReadVariableOp8sequential_144_dense_144_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_144/dense_144/BiasAdd/ReadVariableOp?
 sequential_144/dense_144/BiasAddBiasAdd)sequential_144/dense_144/MatMul:product:07sequential_144/dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_144/dense_144/BiasAdd?
 sequential_144/dense_144/SigmoidSigmoid)sequential_144/dense_144/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_144/dense_144/Sigmoid?
Csequential_144/dense_144/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices?
1sequential_144/dense_144/ActivityRegularizer/MeanMean$sequential_144/dense_144/Sigmoid:y:0Lsequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_144/dense_144/ActivityRegularizer/Mean?
6sequential_144/dense_144/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_144/dense_144/ActivityRegularizer/Maximum/y?
4sequential_144/dense_144/ActivityRegularizer/MaximumMaximum:sequential_144/dense_144/ActivityRegularizer/Mean:output:0?sequential_144/dense_144/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_144/dense_144/ActivityRegularizer/Maximum?
6sequential_144/dense_144/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_144/dense_144/ActivityRegularizer/truediv/x?
4sequential_144/dense_144/ActivityRegularizer/truedivRealDiv?sequential_144/dense_144/ActivityRegularizer/truediv/x:output:08sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_144/dense_144/ActivityRegularizer/truediv?
0sequential_144/dense_144/ActivityRegularizer/LogLog8sequential_144/dense_144/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/Log?
2sequential_144/dense_144/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_144/dense_144/ActivityRegularizer/mul/x?
0sequential_144/dense_144/ActivityRegularizer/mulMul;sequential_144/dense_144/ActivityRegularizer/mul/x:output:04sequential_144/dense_144/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/mul?
2sequential_144/dense_144/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_144/dense_144/ActivityRegularizer/sub/x?
0sequential_144/dense_144/ActivityRegularizer/subSub;sequential_144/dense_144/ActivityRegularizer/sub/x:output:08sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/sub?
8sequential_144/dense_144/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_144/dense_144/ActivityRegularizer/truediv_1/x?
6sequential_144/dense_144/ActivityRegularizer/truediv_1RealDivAsequential_144/dense_144/ActivityRegularizer/truediv_1/x:output:04sequential_144/dense_144/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_144/dense_144/ActivityRegularizer/truediv_1?
2sequential_144/dense_144/ActivityRegularizer/Log_1Log:sequential_144/dense_144/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/Log_1?
4sequential_144/dense_144/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_144/dense_144/ActivityRegularizer/mul_1/x?
2sequential_144/dense_144/ActivityRegularizer/mul_1Mul=sequential_144/dense_144/ActivityRegularizer/mul_1/x:output:06sequential_144/dense_144/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/mul_1?
0sequential_144/dense_144/ActivityRegularizer/addAddV24sequential_144/dense_144/ActivityRegularizer/mul:z:06sequential_144/dense_144/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/add?
2sequential_144/dense_144/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_144/dense_144/ActivityRegularizer/Const?
0sequential_144/dense_144/ActivityRegularizer/SumSum4sequential_144/dense_144/ActivityRegularizer/add:z:0;sequential_144/dense_144/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/Sum?
4sequential_144/dense_144/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_144/dense_144/ActivityRegularizer/mul_2/x?
2sequential_144/dense_144/ActivityRegularizer/mul_2Mul=sequential_144/dense_144/ActivityRegularizer/mul_2/x:output:09sequential_144/dense_144/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/mul_2?
2sequential_144/dense_144/ActivityRegularizer/ShapeShape$sequential_144/dense_144/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_144/dense_144/ActivityRegularizer/Shape?
@sequential_144/dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_144/dense_144/ActivityRegularizer/strided_slice/stack?
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1?
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2?
:sequential_144/dense_144/ActivityRegularizer/strided_sliceStridedSlice;sequential_144/dense_144/ActivityRegularizer/Shape:output:0Isequential_144/dense_144/ActivityRegularizer/strided_slice/stack:output:0Ksequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_144/dense_144/ActivityRegularizer/strided_slice?
1sequential_144/dense_144/ActivityRegularizer/CastCastCsequential_144/dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_144/dense_144/ActivityRegularizer/Cast?
6sequential_144/dense_144/ActivityRegularizer/truediv_2RealDiv6sequential_144/dense_144/ActivityRegularizer/mul_2:z:05sequential_144/dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_144/dense_144/ActivityRegularizer/truediv_2?
.sequential_145/dense_145/MatMul/ReadVariableOpReadVariableOp7sequential_145_dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_145/dense_145/MatMul/ReadVariableOp?
sequential_145/dense_145/MatMulMatMul$sequential_144/dense_144/Sigmoid:y:06sequential_145/dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_145/dense_145/MatMul?
/sequential_145/dense_145/BiasAdd/ReadVariableOpReadVariableOp8sequential_145_dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_145/dense_145/BiasAdd/ReadVariableOp?
 sequential_145/dense_145/BiasAddBiasAdd)sequential_145/dense_145/MatMul:product:07sequential_145/dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_145/dense_145/BiasAdd?
 sequential_145/dense_145/SigmoidSigmoid)sequential_145/dense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_145/dense_145/Sigmoid?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_144_dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_145_dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity$sequential_145/dense_145/Sigmoid:y:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp0^sequential_144/dense_144/BiasAdd/ReadVariableOp/^sequential_144/dense_144/MatMul/ReadVariableOp0^sequential_145/dense_145/BiasAdd/ReadVariableOp/^sequential_145/dense_145/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_144/dense_144/ActivityRegularizer/truediv_2:z:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp0^sequential_144/dense_144/BiasAdd/ReadVariableOp/^sequential_144/dense_144/MatMul/ReadVariableOp0^sequential_145/dense_145/BiasAdd/ReadVariableOp/^sequential_145/dense_145/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_144/dense_144/BiasAdd/ReadVariableOp/sequential_144/dense_144/BiasAdd/ReadVariableOp2`
.sequential_144/dense_144/MatMul/ReadVariableOp.sequential_144/dense_144/MatMul/ReadVariableOp2b
/sequential_145/dense_145/BiasAdd/ReadVariableOp/sequential_145/dense_145/BiasAdd/ReadVariableOp2`
.sequential_145/dense_145/MatMul/ReadVariableOp.sequential_145/dense_145/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?h
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666435
xI
7sequential_144_dense_144_matmul_readvariableop_resource:^ F
8sequential_144_dense_144_biasadd_readvariableop_resource: I
7sequential_145_dense_145_matmul_readvariableop_resource: ^F
8sequential_145_dense_145_biasadd_readvariableop_resource:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?/sequential_144/dense_144/BiasAdd/ReadVariableOp?.sequential_144/dense_144/MatMul/ReadVariableOp?/sequential_145/dense_145/BiasAdd/ReadVariableOp?.sequential_145/dense_145/MatMul/ReadVariableOp?
.sequential_144/dense_144/MatMul/ReadVariableOpReadVariableOp7sequential_144_dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_144/dense_144/MatMul/ReadVariableOp?
sequential_144/dense_144/MatMulMatMulx6sequential_144/dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_144/dense_144/MatMul?
/sequential_144/dense_144/BiasAdd/ReadVariableOpReadVariableOp8sequential_144_dense_144_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_144/dense_144/BiasAdd/ReadVariableOp?
 sequential_144/dense_144/BiasAddBiasAdd)sequential_144/dense_144/MatMul:product:07sequential_144/dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_144/dense_144/BiasAdd?
 sequential_144/dense_144/SigmoidSigmoid)sequential_144/dense_144/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_144/dense_144/Sigmoid?
Csequential_144/dense_144/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices?
1sequential_144/dense_144/ActivityRegularizer/MeanMean$sequential_144/dense_144/Sigmoid:y:0Lsequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_144/dense_144/ActivityRegularizer/Mean?
6sequential_144/dense_144/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_144/dense_144/ActivityRegularizer/Maximum/y?
4sequential_144/dense_144/ActivityRegularizer/MaximumMaximum:sequential_144/dense_144/ActivityRegularizer/Mean:output:0?sequential_144/dense_144/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_144/dense_144/ActivityRegularizer/Maximum?
6sequential_144/dense_144/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_144/dense_144/ActivityRegularizer/truediv/x?
4sequential_144/dense_144/ActivityRegularizer/truedivRealDiv?sequential_144/dense_144/ActivityRegularizer/truediv/x:output:08sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_144/dense_144/ActivityRegularizer/truediv?
0sequential_144/dense_144/ActivityRegularizer/LogLog8sequential_144/dense_144/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/Log?
2sequential_144/dense_144/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_144/dense_144/ActivityRegularizer/mul/x?
0sequential_144/dense_144/ActivityRegularizer/mulMul;sequential_144/dense_144/ActivityRegularizer/mul/x:output:04sequential_144/dense_144/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/mul?
2sequential_144/dense_144/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_144/dense_144/ActivityRegularizer/sub/x?
0sequential_144/dense_144/ActivityRegularizer/subSub;sequential_144/dense_144/ActivityRegularizer/sub/x:output:08sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/sub?
8sequential_144/dense_144/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_144/dense_144/ActivityRegularizer/truediv_1/x?
6sequential_144/dense_144/ActivityRegularizer/truediv_1RealDivAsequential_144/dense_144/ActivityRegularizer/truediv_1/x:output:04sequential_144/dense_144/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_144/dense_144/ActivityRegularizer/truediv_1?
2sequential_144/dense_144/ActivityRegularizer/Log_1Log:sequential_144/dense_144/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/Log_1?
4sequential_144/dense_144/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_144/dense_144/ActivityRegularizer/mul_1/x?
2sequential_144/dense_144/ActivityRegularizer/mul_1Mul=sequential_144/dense_144/ActivityRegularizer/mul_1/x:output:06sequential_144/dense_144/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/mul_1?
0sequential_144/dense_144/ActivityRegularizer/addAddV24sequential_144/dense_144/ActivityRegularizer/mul:z:06sequential_144/dense_144/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/add?
2sequential_144/dense_144/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_144/dense_144/ActivityRegularizer/Const?
0sequential_144/dense_144/ActivityRegularizer/SumSum4sequential_144/dense_144/ActivityRegularizer/add:z:0;sequential_144/dense_144/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_144/dense_144/ActivityRegularizer/Sum?
4sequential_144/dense_144/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_144/dense_144/ActivityRegularizer/mul_2/x?
2sequential_144/dense_144/ActivityRegularizer/mul_2Mul=sequential_144/dense_144/ActivityRegularizer/mul_2/x:output:09sequential_144/dense_144/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_144/dense_144/ActivityRegularizer/mul_2?
2sequential_144/dense_144/ActivityRegularizer/ShapeShape$sequential_144/dense_144/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_144/dense_144/ActivityRegularizer/Shape?
@sequential_144/dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_144/dense_144/ActivityRegularizer/strided_slice/stack?
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1?
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2?
:sequential_144/dense_144/ActivityRegularizer/strided_sliceStridedSlice;sequential_144/dense_144/ActivityRegularizer/Shape:output:0Isequential_144/dense_144/ActivityRegularizer/strided_slice/stack:output:0Ksequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_144/dense_144/ActivityRegularizer/strided_slice?
1sequential_144/dense_144/ActivityRegularizer/CastCastCsequential_144/dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_144/dense_144/ActivityRegularizer/Cast?
6sequential_144/dense_144/ActivityRegularizer/truediv_2RealDiv6sequential_144/dense_144/ActivityRegularizer/mul_2:z:05sequential_144/dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_144/dense_144/ActivityRegularizer/truediv_2?
.sequential_145/dense_145/MatMul/ReadVariableOpReadVariableOp7sequential_145_dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_145/dense_145/MatMul/ReadVariableOp?
sequential_145/dense_145/MatMulMatMul$sequential_144/dense_144/Sigmoid:y:06sequential_145/dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_145/dense_145/MatMul?
/sequential_145/dense_145/BiasAdd/ReadVariableOpReadVariableOp8sequential_145_dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_145/dense_145/BiasAdd/ReadVariableOp?
 sequential_145/dense_145/BiasAddBiasAdd)sequential_145/dense_145/MatMul:product:07sequential_145/dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_145/dense_145/BiasAdd?
 sequential_145/dense_145/SigmoidSigmoid)sequential_145/dense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_145/dense_145/Sigmoid?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_144_dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_145_dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity$sequential_145/dense_145/Sigmoid:y:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp0^sequential_144/dense_144/BiasAdd/ReadVariableOp/^sequential_144/dense_144/MatMul/ReadVariableOp0^sequential_145/dense_145/BiasAdd/ReadVariableOp/^sequential_145/dense_145/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_144/dense_144/ActivityRegularizer/truediv_2:z:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp0^sequential_144/dense_144/BiasAdd/ReadVariableOp/^sequential_144/dense_144/MatMul/ReadVariableOp0^sequential_145/dense_145/BiasAdd/ReadVariableOp/^sequential_145/dense_145/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_144/dense_144/BiasAdd/ReadVariableOp/sequential_144/dense_144/BiasAdd/ReadVariableOp2`
.sequential_144/dense_144/MatMul/ReadVariableOp.sequential_144/dense_144/MatMul/ReadVariableOp2b
/sequential_145/dense_145/BiasAdd/ReadVariableOp/sequential_145/dense_145/BiasAdd/ReadVariableOp2`
.sequential_145/dense_145/MatMul/ReadVariableOp.sequential_145/dense_145/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
__inference_loss_fn_1_16666743M
;dense_145_kernel_regularizer_square_readvariableop_resource: ^
identity??2dense_145/kernel/Regularizer/Square/ReadVariableOp?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_145_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity$dense_145/kernel/Regularizer/mul:z:03^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp
?#
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665942
input_73$
dense_144_16665921:^  
dense_144_16665923: 
identity

identity_1??!dense_144/StatefulPartitionedCall?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
!dense_144/StatefulPartitionedCallStatefulPartitionedCallinput_73dense_144_16665921dense_144_16665923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122#
!dense_144/StatefulPartitionedCall?
-dense_144/ActivityRegularizer/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_144_activity_regularizer_166657882/
-dense_144/ActivityRegularizer/PartitionedCall?
#dense_144/ActivityRegularizer/ShapeShape*dense_144/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
%dense_144/ActivityRegularizer/truedivRealDiv6dense_144/ActivityRegularizer/PartitionedCall:output:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_144_16665921*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentity*dense_144/StatefulPartitionedCall:output:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_144/ActivityRegularizer/truediv:z:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_73
?#
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665834

inputs$
dense_144_16665813:^  
dense_144_16665815: 
identity

identity_1??!dense_144/StatefulPartitionedCall?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
!dense_144/StatefulPartitionedCallStatefulPartitionedCallinputsdense_144_16665813dense_144_16665815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122#
!dense_144/StatefulPartitionedCall?
-dense_144/ActivityRegularizer/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_144_activity_regularizer_166657882/
-dense_144/ActivityRegularizer/PartitionedCall?
#dense_144/ActivityRegularizer/ShapeShape*dense_144/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
%dense_144/ActivityRegularizer/truedivRealDiv6dense_144/ActivityRegularizer/PartitionedCall:output:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_144_16665813*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentity*dense_144/StatefulPartitionedCall:output:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_144/ActivityRegularizer/truediv:z:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666180
x)
sequential_144_16666155:^ %
sequential_144_16666157: )
sequential_145_16666161: ^%
sequential_145_16666163:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?&sequential_144/StatefulPartitionedCall?&sequential_145/StatefulPartitionedCall?
&sequential_144/StatefulPartitionedCallStatefulPartitionedCallxsequential_144_16666155sequential_144_16666157*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166659002(
&sequential_144/StatefulPartitionedCall?
&sequential_145/StatefulPartitionedCallStatefulPartitionedCall/sequential_144/StatefulPartitionedCall:output:0sequential_145_16666161sequential_145_16666163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660462(
&sequential_145/StatefulPartitionedCall?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_144_16666155*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_145_16666161*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity/sequential_145/StatefulPartitionedCall:output:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_144/StatefulPartitionedCall:output:13^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_144/StatefulPartitionedCall&sequential_144/StatefulPartitionedCall2P
&sequential_145/StatefulPartitionedCall&sequential_145/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?_
?
#__inference__wrapped_model_16665759
input_1X
Fautoencoder_72_sequential_144_dense_144_matmul_readvariableop_resource:^ U
Gautoencoder_72_sequential_144_dense_144_biasadd_readvariableop_resource: X
Fautoencoder_72_sequential_145_dense_145_matmul_readvariableop_resource: ^U
Gautoencoder_72_sequential_145_dense_145_biasadd_readvariableop_resource:^
identity??>autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp?=autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp?>autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp?=autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp?
=autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOpReadVariableOpFautoencoder_72_sequential_144_dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02?
=autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp?
.autoencoder_72/sequential_144/dense_144/MatMulMatMulinput_1Eautoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.autoencoder_72/sequential_144/dense_144/MatMul?
>autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_72_sequential_144_dense_144_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp?
/autoencoder_72/sequential_144/dense_144/BiasAddBiasAdd8autoencoder_72/sequential_144/dense_144/MatMul:product:0Fautoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_72/sequential_144/dense_144/BiasAdd?
/autoencoder_72/sequential_144/dense_144/SigmoidSigmoid8autoencoder_72/sequential_144/dense_144/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_72/sequential_144/dense_144/Sigmoid?
Rautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices?
@autoencoder_72/sequential_144/dense_144/ActivityRegularizer/MeanMean3autoencoder_72/sequential_144/dense_144/Sigmoid:y:0[autoencoder_72/sequential_144/dense_144/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2B
@autoencoder_72/sequential_144/dense_144/ActivityRegularizer/Mean?
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2G
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum/y?
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/MaximumMaximumIautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Mean:output:0Nautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2E
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum?
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2G
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv/x?
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truedivRealDivNautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv/x:output:0Gautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv?
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/LogLogGautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2A
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/Log?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul/x?
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/mulMulJautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul/x:output:0Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2A
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/sub/x?
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/subSubJautoencoder_72/sequential_144/dense_144/ActivityRegularizer/sub/x:output:0Gautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2A
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/sub?
Gautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2I
Gautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1/x?
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1RealDivPautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1/x:output:0Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2G
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Log_1LogIautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Log_1?
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2E
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1/x?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1MulLautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1/x:output:0Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1?
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/addAddV2Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul:z:0Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/add?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Const?
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/SumSumCautoencoder_72/sequential_144/dense_144/ActivityRegularizer/add:z:0Jautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2A
?autoencoder_72/sequential_144/dense_144/ActivityRegularizer/Sum?
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
Cautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2/x?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2MulLautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2/x:output:0Hautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2?
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/ShapeShape3autoencoder_72/sequential_144/dense_144/Sigmoid:y:0*
T0*
_output_shapes
:2C
Aautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Shape?
Oautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack?
Qautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1?
Qautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2?
Iautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_sliceStridedSliceJautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Shape:output:0Xautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack:output:0Zautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_1:output:0Zautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice?
@autoencoder_72/sequential_144/dense_144/ActivityRegularizer/CastCastRautoencoder_72/sequential_144/dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@autoencoder_72/sequential_144/dense_144/ActivityRegularizer/Cast?
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_2RealDivEautoencoder_72/sequential_144/dense_144/ActivityRegularizer/mul_2:z:0Dautoencoder_72/sequential_144/dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2G
Eautoencoder_72/sequential_144/dense_144/ActivityRegularizer/truediv_2?
=autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOpReadVariableOpFautoencoder_72_sequential_145_dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02?
=autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp?
.autoencoder_72/sequential_145/dense_145/MatMulMatMul3autoencoder_72/sequential_144/dense_144/Sigmoid:y:0Eautoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^20
.autoencoder_72/sequential_145/dense_145/MatMul?
>autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_72_sequential_145_dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02@
>autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp?
/autoencoder_72/sequential_145/dense_145/BiasAddBiasAdd8autoencoder_72/sequential_145/dense_145/MatMul:product:0Fautoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_72/sequential_145/dense_145/BiasAdd?
/autoencoder_72/sequential_145/dense_145/SigmoidSigmoid8autoencoder_72/sequential_145/dense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_72/sequential_145/dense_145/Sigmoid?
IdentityIdentity3autoencoder_72/sequential_145/dense_145/Sigmoid:y:0?^autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp>^autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp?^autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp>^autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2?
>autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp>autoencoder_72/sequential_144/dense_144/BiasAdd/ReadVariableOp2~
=autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp=autoencoder_72/sequential_144/dense_144/MatMul/ReadVariableOp2?
>autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp>autoencoder_72/sequential_145/dense_145/BiasAdd/ReadVariableOp2~
=autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp=autoencoder_72/sequential_145/dense_145/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?%
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666124
x)
sequential_144_16666099:^ %
sequential_144_16666101: )
sequential_145_16666105: ^%
sequential_145_16666107:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?&sequential_144/StatefulPartitionedCall?&sequential_145/StatefulPartitionedCall?
&sequential_144/StatefulPartitionedCallStatefulPartitionedCallxsequential_144_16666099sequential_144_16666101*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166658342(
&sequential_144/StatefulPartitionedCall?
&sequential_145/StatefulPartitionedCallStatefulPartitionedCall/sequential_144/StatefulPartitionedCall:output:0sequential_145_16666105sequential_145_16666107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660032(
&sequential_145/StatefulPartitionedCall?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_144_16666099*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_145_16666105*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity/sequential_145/StatefulPartitionedCall:output:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_144/StatefulPartitionedCall:output:13^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_144/StatefulPartitionedCall&sequential_144/StatefulPartitionedCall2P
&sequential_145/StatefulPartitionedCall&sequential_145/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_145_layer_call_fn_16666568
dense_145_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_145_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_145_input
?
?
1__inference_sequential_144_layer_call_fn_16666451

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166658342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_145_layer_call_fn_16666595
dense_145_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_145_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_145_input
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666003

inputs$
dense_145_16665991: ^ 
dense_145_16665993:^
identity??!dense_145/StatefulPartitionedCall?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
!dense_145/StatefulPartitionedCallStatefulPartitionedCallinputsdense_145_16665991dense_145_16665993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_145_layer_call_and_return_conditional_losses_166659902#
!dense_145/StatefulPartitionedCall?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_145_16665991*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity*dense_145/StatefulPartitionedCall:output:0"^dense_145/StatefulPartitionedCall3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_145/StatefulPartitionedCall!dense_145/StatefulPartitionedCall2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_72_layer_call_fn_16666303
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_166661242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_144_layer_call_fn_16665918
input_73
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_73unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166659002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_73
?
?
1__inference_sequential_145_layer_call_fn_16666586

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_16666289
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_166657592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
S
3__inference_dense_144_activity_regularizer_16665788

activation
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesd
MeanMean
activationMean/reduction_indices:output:0*
T0*
_output_shapes
:2
Mean[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
	Maximum/yc
MaximumMaximumMean:output:0Maximum/y:output:0*
T0*
_output_shapes
:2	
Maximum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
	truediv/xa
truedivRealDivtruediv/x:output:0Maximum:z:0*
T0*
_output_shapes
:2	
truedivA
LogLogtruediv:z:0*
T0*
_output_shapes
:2
LogS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xM
mulMulmul/x:output:0Log:y:0*
T0*
_output_shapes
:2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xQ
subSubsub/x:output:0Maximum:z:0*
T0*
_output_shapes
:2
sub_
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
truediv_1/xc
	truediv_1RealDivtruediv_1/x:output:0sub:z:0*
T0*
_output_shapes
:2
	truediv_1G
Log_1Logtruediv_1:z:0*
T0*
_output_shapes
:2
Log_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
mul_1/xU
mul_1Mulmul_1/x:output:0	Log_1:y:0*
T0*
_output_shapes
:2
mul_1J
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add>
RankRankadd:z:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeK
SumSumadd:z:0range:output:0*
T0*
_output_shapes
: 2
SumW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xV
mul_2Mulmul_2/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mul_2L
IdentityIdentity	mul_2:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::D @

_output_shapes
:
$
_user_specified_name
activation
?
?
1__inference_sequential_144_layer_call_fn_16665842
input_73
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_73unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166658342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_73
?%
?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666262
input_1)
sequential_144_16666237:^ %
sequential_144_16666239: )
sequential_145_16666243: ^%
sequential_145_16666245:^
identity

identity_1??2dense_144/kernel/Regularizer/Square/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?&sequential_144/StatefulPartitionedCall?&sequential_145/StatefulPartitionedCall?
&sequential_144/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_144_16666237sequential_144_16666239*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166659002(
&sequential_144/StatefulPartitionedCall?
&sequential_145/StatefulPartitionedCallStatefulPartitionedCall/sequential_144/StatefulPartitionedCall:output:0sequential_145_16666243sequential_145_16666245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660462(
&sequential_145/StatefulPartitionedCall?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_144_16666237*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_145_16666243*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentity/sequential_145/StatefulPartitionedCall:output:03^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_144/StatefulPartitionedCall:output:13^dense_144/kernel/Regularizer/Square/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp'^sequential_144/StatefulPartitionedCall'^sequential_145/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_144/StatefulPartitionedCall&sequential_144/StatefulPartitionedCall2P
&sequential_145/StatefulPartitionedCall&sequential_145/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666612

inputs:
(dense_145_matmul_readvariableop_resource: ^7
)dense_145_biasadd_readvariableop_resource:^
identity?? dense_145/BiasAdd/ReadVariableOp?dense_145/MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
dense_145/MatMul/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_145/MatMul/ReadVariableOp?
dense_145/MatMulMatMulinputs'dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/MatMul?
 dense_145/BiasAdd/ReadVariableOpReadVariableOp)dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_145/BiasAdd/ReadVariableOp?
dense_145/BiasAddBiasAdddense_145/MatMul:product:0(dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/BiasAdd
dense_145/SigmoidSigmoiddense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_145/Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitydense_145/Sigmoid:y:0!^dense_145/BiasAdd/ReadVariableOp ^dense_145/MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_145/BiasAdd/ReadVariableOp dense_145/BiasAdd/ReadVariableOp2B
dense_145/MatMul/ReadVariableOpdense_145/MatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666663
dense_145_input:
(dense_145_matmul_readvariableop_resource: ^7
)dense_145_biasadd_readvariableop_resource:^
identity?? dense_145/BiasAdd/ReadVariableOp?dense_145/MatMul/ReadVariableOp?2dense_145/kernel/Regularizer/Square/ReadVariableOp?
dense_145/MatMul/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_145/MatMul/ReadVariableOp?
dense_145/MatMulMatMuldense_145_input'dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/MatMul?
 dense_145/BiasAdd/ReadVariableOpReadVariableOp)dense_145_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_145/BiasAdd/ReadVariableOp?
dense_145/BiasAddBiasAdddense_145/MatMul:product:0(dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_145/BiasAdd
dense_145/SigmoidSigmoiddense_145/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_145/Sigmoid?
2dense_145/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_145/kernel/Regularizer/Square/ReadVariableOp?
#dense_145/kernel/Regularizer/SquareSquare:dense_145/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_145/kernel/Regularizer/Square?
"dense_145/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_145/kernel/Regularizer/Const?
 dense_145/kernel/Regularizer/SumSum'dense_145/kernel/Regularizer/Square:y:0+dense_145/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/Sum?
"dense_145/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_145/kernel/Regularizer/mul/x?
 dense_145/kernel/Regularizer/mulMul+dense_145/kernel/Regularizer/mul/x:output:0)dense_145/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_145/kernel/Regularizer/mul?
IdentityIdentitydense_145/Sigmoid:y:0!^dense_145/BiasAdd/ReadVariableOp ^dense_145/MatMul/ReadVariableOp3^dense_145/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_145/BiasAdd/ReadVariableOp dense_145/BiasAdd/ReadVariableOp2B
dense_145/MatMul/ReadVariableOpdense_145/MatMul/ReadVariableOp2h
2dense_145/kernel/Regularizer/Square/ReadVariableOp2dense_145/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_145_input
?B
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666507

inputs:
(dense_144_matmul_readvariableop_resource:^ 7
)dense_144_biasadd_readvariableop_resource: 
identity

identity_1?? dense_144/BiasAdd/ReadVariableOp?dense_144/MatMul/ReadVariableOp?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
dense_144/MatMul/ReadVariableOpReadVariableOp(dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_144/MatMul/ReadVariableOp?
dense_144/MatMulMatMulinputs'dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_144/MatMul?
 dense_144/BiasAdd/ReadVariableOpReadVariableOp)dense_144_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_144/BiasAdd/ReadVariableOp?
dense_144/BiasAddBiasAdddense_144/MatMul:product:0(dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_144/BiasAdd
dense_144/SigmoidSigmoiddense_144/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_144/Sigmoid?
4dense_144/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_144/ActivityRegularizer/Mean/reduction_indices?
"dense_144/ActivityRegularizer/MeanMeandense_144/Sigmoid:y:0=dense_144/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Mean?
'dense_144/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_144/ActivityRegularizer/Maximum/y?
%dense_144/ActivityRegularizer/MaximumMaximum+dense_144/ActivityRegularizer/Mean:output:00dense_144/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/Maximum?
'dense_144/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_144/ActivityRegularizer/truediv/x?
%dense_144/ActivityRegularizer/truedivRealDiv0dense_144/ActivityRegularizer/truediv/x:output:0)dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
!dense_144/ActivityRegularizer/LogLog)dense_144/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/Log?
#dense_144/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_144/ActivityRegularizer/mul/x?
!dense_144/ActivityRegularizer/mulMul,dense_144/ActivityRegularizer/mul/x:output:0%dense_144/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/mul?
#dense_144/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_144/ActivityRegularizer/sub/x?
!dense_144/ActivityRegularizer/subSub,dense_144/ActivityRegularizer/sub/x:output:0)dense_144/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/sub?
)dense_144/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_144/ActivityRegularizer/truediv_1/x?
'dense_144/ActivityRegularizer/truediv_1RealDiv2dense_144/ActivityRegularizer/truediv_1/x:output:0%dense_144/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_144/ActivityRegularizer/truediv_1?
#dense_144/ActivityRegularizer/Log_1Log+dense_144/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/Log_1?
%dense_144/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_144/ActivityRegularizer/mul_1/x?
#dense_144/ActivityRegularizer/mul_1Mul.dense_144/ActivityRegularizer/mul_1/x:output:0'dense_144/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/mul_1?
!dense_144/ActivityRegularizer/addAddV2%dense_144/ActivityRegularizer/mul:z:0'dense_144/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/add?
#dense_144/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_144/ActivityRegularizer/Const?
!dense_144/ActivityRegularizer/SumSum%dense_144/ActivityRegularizer/add:z:0,dense_144/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_144/ActivityRegularizer/Sum?
%dense_144/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_144/ActivityRegularizer/mul_2/x?
#dense_144/ActivityRegularizer/mul_2Mul.dense_144/ActivityRegularizer/mul_2/x:output:0*dense_144/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_144/ActivityRegularizer/mul_2?
#dense_144/ActivityRegularizer/ShapeShapedense_144/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
'dense_144/ActivityRegularizer/truediv_2RealDiv'dense_144/ActivityRegularizer/mul_2:z:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_144/ActivityRegularizer/truediv_2?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_144_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentitydense_144/Sigmoid:y:0!^dense_144/BiasAdd/ReadVariableOp ^dense_144/MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_144/ActivityRegularizer/truediv_2:z:0!^dense_144/BiasAdd/ReadVariableOp ^dense_144/MatMul/ReadVariableOp3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_144/BiasAdd/ReadVariableOp dense_144/BiasAdd/ReadVariableOp2B
dense_144/MatMul/ReadVariableOpdense_144/MatMul/ReadVariableOp2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
,__inference_dense_144_layer_call_fn_16666678

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_16666700M
;dense_144_kernel_regularizer_square_readvariableop_resource:^ 
identity??2dense_144/kernel/Regularizer/Square/ReadVariableOp?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_144_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentity$dense_144/kernel/Regularizer/mul:z:03^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp
?
?
,__inference_dense_145_layer_call_fn_16666715

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_145_layer_call_and_return_conditional_losses_166659902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_145_layer_call_fn_16666577

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_145_layer_call_and_return_conditional_losses_166660032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_144_layer_call_fn_16666461

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_144_layer_call_and_return_conditional_losses_166659002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_72_layer_call_fn_16666136
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_166661242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?#
?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665900

inputs$
dense_144_16665879:^  
dense_144_16665881: 
identity

identity_1??!dense_144/StatefulPartitionedCall?2dense_144/kernel/Regularizer/Square/ReadVariableOp?
!dense_144/StatefulPartitionedCallStatefulPartitionedCallinputsdense_144_16665879dense_144_16665881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_144_layer_call_and_return_conditional_losses_166658122#
!dense_144/StatefulPartitionedCall?
-dense_144/ActivityRegularizer/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *<
f7R5
3__inference_dense_144_activity_regularizer_166657882/
-dense_144/ActivityRegularizer/PartitionedCall?
#dense_144/ActivityRegularizer/ShapeShape*dense_144/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_144/ActivityRegularizer/Shape?
1dense_144/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_144/ActivityRegularizer/strided_slice/stack?
3dense_144/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_1?
3dense_144/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_144/ActivityRegularizer/strided_slice/stack_2?
+dense_144/ActivityRegularizer/strided_sliceStridedSlice,dense_144/ActivityRegularizer/Shape:output:0:dense_144/ActivityRegularizer/strided_slice/stack:output:0<dense_144/ActivityRegularizer/strided_slice/stack_1:output:0<dense_144/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_144/ActivityRegularizer/strided_slice?
"dense_144/ActivityRegularizer/CastCast4dense_144/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_144/ActivityRegularizer/Cast?
%dense_144/ActivityRegularizer/truedivRealDiv6dense_144/ActivityRegularizer/PartitionedCall:output:0&dense_144/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_144/ActivityRegularizer/truediv?
2dense_144/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_144_16665879*
_output_shapes

:^ *
dtype024
2dense_144/kernel/Regularizer/Square/ReadVariableOp?
#dense_144/kernel/Regularizer/SquareSquare:dense_144/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_144/kernel/Regularizer/Square?
"dense_144/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_144/kernel/Regularizer/Const?
 dense_144/kernel/Regularizer/SumSum'dense_144/kernel/Regularizer/Square:y:0+dense_144/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/Sum?
"dense_144/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_144/kernel/Regularizer/mul/x?
 dense_144/kernel/Regularizer/mulMul+dense_144/kernel/Regularizer/mul/x:output:0)dense_144/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_144/kernel/Regularizer/mul?
IdentityIdentity*dense_144/StatefulPartitionedCall:output:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_144/ActivityRegularizer/truediv:z:0"^dense_144/StatefulPartitionedCall3^dense_144/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2h
2dense_144/kernel/Regularizer/Square/ReadVariableOp2dense_144/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????^<
output_10
StatefulPartitionedCall:0?????????^tensorflow/serving/predict:??
?
history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "autoencoder_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
?
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_144", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_144", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_73"}}, {"class_name": "Dense", "config": {"name": "dense_144", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_73"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_144", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_73"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_144", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_145", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_145", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_145_input"}}, {"class_name": "Dense", "config": {"name": "dense_145", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_145_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_145", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_145_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_145", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_144", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_144", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
?
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?	

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_145", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_145", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
?
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
": ^ 2dense_144/kernel
: 2dense_144/bias
":  ^2dense_145/kernel
:^2dense_145/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
?
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
@__call__
Factivity_regularizer_fn
*A&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
?
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
1__inference_autoencoder_72_layer_call_fn_16666136
1__inference_autoencoder_72_layer_call_fn_16666303
1__inference_autoencoder_72_layer_call_fn_16666317
1__inference_autoencoder_72_layer_call_fn_16666206?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference__wrapped_model_16665759?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????^
?2?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666376
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666435
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666234
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666262?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_sequential_144_layer_call_fn_16665842
1__inference_sequential_144_layer_call_fn_16666451
1__inference_sequential_144_layer_call_fn_16666461
1__inference_sequential_144_layer_call_fn_16665918?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666507
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666553
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665942
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665966?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_sequential_145_layer_call_fn_16666568
1__inference_sequential_145_layer_call_fn_16666577
1__inference_sequential_145_layer_call_fn_16666586
1__inference_sequential_145_layer_call_fn_16666595?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666612
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666629
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666646
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666663?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_16666289input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_144_layer_call_fn_16666678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_dense_144_layer_call_and_return_all_conditional_losses_16666689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_16666700?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
,__inference_dense_145_layer_call_fn_16666715?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_145_layer_call_and_return_conditional_losses_16666732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_1_16666743?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
3__inference_dense_144_activity_regularizer_16665788?
???
FullArgSpec!
args?
jself
j
activation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
G__inference_dense_144_layer_call_and_return_conditional_losses_16666760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_16665759m0?-
&?#
!?
input_1?????????^
? "3?0
.
output_1"?
output_1?????????^?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666234q4?1
*?'
!?
input_1?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666262q4?1
*?'
!?
input_1?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666376k.?+
$?!
?
X?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_72_layer_call_and_return_conditional_losses_16666435k.?+
$?!
?
X?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
1__inference_autoencoder_72_layer_call_fn_16666136V4?1
*?'
!?
input_1?????????^
p 
? "??????????^?
1__inference_autoencoder_72_layer_call_fn_16666206V4?1
*?'
!?
input_1?????????^
p
? "??????????^?
1__inference_autoencoder_72_layer_call_fn_16666303P.?+
$?!
?
X?????????^
p 
? "??????????^?
1__inference_autoencoder_72_layer_call_fn_16666317P.?+
$?!
?
X?????????^
p
? "??????????^f
3__inference_dense_144_activity_regularizer_16665788/$?!
?
?

activation
? "? ?
K__inference_dense_144_layer_call_and_return_all_conditional_losses_16666689j/?,
%?"
 ?
inputs?????????^
? "3?0
?
0????????? 
?
?	
1/0 ?
G__inference_dense_144_layer_call_and_return_conditional_losses_16666760\/?,
%?"
 ?
inputs?????????^
? "%?"
?
0????????? 
? 
,__inference_dense_144_layer_call_fn_16666678O/?,
%?"
 ?
inputs?????????^
? "?????????? ?
G__inference_dense_145_layer_call_and_return_conditional_losses_16666732\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????^
? 
,__inference_dense_145_layer_call_fn_16666715O/?,
%?"
 ?
inputs????????? 
? "??????????^=
__inference_loss_fn_0_16666700?

? 
? "? =
__inference_loss_fn_1_16666743?

? 
? "? ?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665942t9?6
/?,
"?
input_73?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16665966t9?6
/?,
"?
input_73?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666507r7?4
-?*
 ?
inputs?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_144_layer_call_and_return_conditional_losses_16666553r7?4
-?*
 ?
inputs?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
1__inference_sequential_144_layer_call_fn_16665842Y9?6
/?,
"?
input_73?????????^
p 

 
? "?????????? ?
1__inference_sequential_144_layer_call_fn_16665918Y9?6
/?,
"?
input_73?????????^
p

 
? "?????????? ?
1__inference_sequential_144_layer_call_fn_16666451W7?4
-?*
 ?
inputs?????????^
p 

 
? "?????????? ?
1__inference_sequential_144_layer_call_fn_16666461W7?4
-?*
 ?
inputs?????????^
p

 
? "?????????? ?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666612d7?4
-?*
 ?
inputs????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666629d7?4
-?*
 ?
inputs????????? 
p

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666646m@?=
6?3
)?&
dense_145_input????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_145_layer_call_and_return_conditional_losses_16666663m@?=
6?3
)?&
dense_145_input????????? 
p

 
? "%?"
?
0?????????^
? ?
1__inference_sequential_145_layer_call_fn_16666568`@?=
6?3
)?&
dense_145_input????????? 
p 

 
? "??????????^?
1__inference_sequential_145_layer_call_fn_16666577W7?4
-?*
 ?
inputs????????? 
p 

 
? "??????????^?
1__inference_sequential_145_layer_call_fn_16666586W7?4
-?*
 ?
inputs????????? 
p

 
? "??????????^?
1__inference_sequential_145_layer_call_fn_16666595`@?=
6?3
)?&
dense_145_input????????? 
p

 
? "??????????^?
&__inference_signature_wrapper_16666289x;?8
? 
1?.
,
input_1!?
input_1?????????^"3?0
.
output_1"?
output_1?????????^