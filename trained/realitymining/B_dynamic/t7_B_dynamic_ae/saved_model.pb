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
dense_104/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ *!
shared_namedense_104/kernel
u
$dense_104/kernel/Read/ReadVariableOpReadVariableOpdense_104/kernel*
_output_shapes

:^ *
dtype0
t
dense_104/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_104/bias
m
"dense_104/bias/Read/ReadVariableOpReadVariableOpdense_104/bias*
_output_shapes
: *
dtype0
|
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^*!
shared_namedense_105/kernel
u
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel*
_output_shapes

: ^*
dtype0
t
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_105/bias
m
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
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
VARIABLE_VALUEdense_104/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_104/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_105/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_105/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_104/kerneldense_104/biasdense_105/kerneldense_105/bias*
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
&__inference_signature_wrapper_16641269
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_104/kernel/Read/ReadVariableOp"dense_104/bias/Read/ReadVariableOp$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_16641775
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_104/kerneldense_104/biasdense_105/kerneldense_105/bias*
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
$__inference__traced_restore_16641797??	
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641643
dense_105_input:
(dense_105_matmul_readvariableop_resource: ^7
)dense_105_biasadd_readvariableop_resource:^
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMuldense_105_input'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/BiasAdd
dense_105/SigmoidSigmoiddense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_105/Sigmoid?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitydense_105/Sigmoid:y:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_105_input
?
?
K__inference_dense_104_layer_call_and_return_all_conditional_losses_16641669

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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922
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
3__inference_dense_104_activity_regularizer_166407682
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
?#
?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640922
input_53$
dense_104_16640901:^  
dense_104_16640903: 
identity

identity_1??!dense_104/StatefulPartitionedCall?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinput_53dense_104_16640901dense_104_16640903*
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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922#
!dense_104/StatefulPartitionedCall?
-dense_104/ActivityRegularizer/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
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
3__inference_dense_104_activity_regularizer_166407682/
-dense_104/ActivityRegularizer/PartitionedCall?
#dense_104/ActivityRegularizer/ShapeShape*dense_104/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
%dense_104/ActivityRegularizer/truedivRealDiv6dense_104/ActivityRegularizer/PartitionedCall:output:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_104_16640901*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_104/ActivityRegularizer/truediv:z:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_53
?
?
G__inference_dense_105_layer_call_and_return_conditional_losses_16640970

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_52_layer_call_fn_16641283
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_166411042
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
?B
?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641487

inputs:
(dense_104_matmul_readvariableop_resource:^ 7
)dense_104_biasadd_readvariableop_resource: 
identity

identity_1?? dense_104/BiasAdd/ReadVariableOp?dense_104/MatMul/ReadVariableOp?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_104/MatMul/ReadVariableOp?
dense_104/MatMulMatMulinputs'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_104/MatMul?
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_104/BiasAdd/ReadVariableOp?
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_104/BiasAdd
dense_104/SigmoidSigmoiddense_104/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_104/Sigmoid?
4dense_104/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_104/ActivityRegularizer/Mean/reduction_indices?
"dense_104/ActivityRegularizer/MeanMeandense_104/Sigmoid:y:0=dense_104/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Mean?
'dense_104/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_104/ActivityRegularizer/Maximum/y?
%dense_104/ActivityRegularizer/MaximumMaximum+dense_104/ActivityRegularizer/Mean:output:00dense_104/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/Maximum?
'dense_104/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_104/ActivityRegularizer/truediv/x?
%dense_104/ActivityRegularizer/truedivRealDiv0dense_104/ActivityRegularizer/truediv/x:output:0)dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
!dense_104/ActivityRegularizer/LogLog)dense_104/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/Log?
#dense_104/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_104/ActivityRegularizer/mul/x?
!dense_104/ActivityRegularizer/mulMul,dense_104/ActivityRegularizer/mul/x:output:0%dense_104/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/mul?
#dense_104/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_104/ActivityRegularizer/sub/x?
!dense_104/ActivityRegularizer/subSub,dense_104/ActivityRegularizer/sub/x:output:0)dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/sub?
)dense_104/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_104/ActivityRegularizer/truediv_1/x?
'dense_104/ActivityRegularizer/truediv_1RealDiv2dense_104/ActivityRegularizer/truediv_1/x:output:0%dense_104/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_104/ActivityRegularizer/truediv_1?
#dense_104/ActivityRegularizer/Log_1Log+dense_104/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/Log_1?
%dense_104/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_104/ActivityRegularizer/mul_1/x?
#dense_104/ActivityRegularizer/mul_1Mul.dense_104/ActivityRegularizer/mul_1/x:output:0'dense_104/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/mul_1?
!dense_104/ActivityRegularizer/addAddV2%dense_104/ActivityRegularizer/mul:z:0'dense_104/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/add?
#dense_104/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_104/ActivityRegularizer/Const?
!dense_104/ActivityRegularizer/SumSum%dense_104/ActivityRegularizer/add:z:0,dense_104/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/Sum?
%dense_104/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_104/ActivityRegularizer/mul_2/x?
#dense_104/ActivityRegularizer/mul_2Mul.dense_104/ActivityRegularizer/mul_2/x:output:0*dense_104/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/mul_2?
#dense_104/ActivityRegularizer/ShapeShapedense_104/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
'dense_104/ActivityRegularizer/truediv_2RealDiv'dense_104/ActivityRegularizer/mul_2:z:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_104/ActivityRegularizer/truediv_2?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentitydense_104/Sigmoid:y:0!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_104/ActivityRegularizer/truediv_2:z:0!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_105_layer_call_fn_16641566

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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166410262
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
?h
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641356
xI
7sequential_104_dense_104_matmul_readvariableop_resource:^ F
8sequential_104_dense_104_biasadd_readvariableop_resource: I
7sequential_105_dense_105_matmul_readvariableop_resource: ^F
8sequential_105_dense_105_biasadd_readvariableop_resource:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?/sequential_104/dense_104/BiasAdd/ReadVariableOp?.sequential_104/dense_104/MatMul/ReadVariableOp?/sequential_105/dense_105/BiasAdd/ReadVariableOp?.sequential_105/dense_105/MatMul/ReadVariableOp?
.sequential_104/dense_104/MatMul/ReadVariableOpReadVariableOp7sequential_104_dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_104/dense_104/MatMul/ReadVariableOp?
sequential_104/dense_104/MatMulMatMulx6sequential_104/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_104/dense_104/MatMul?
/sequential_104/dense_104/BiasAdd/ReadVariableOpReadVariableOp8sequential_104_dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_104/dense_104/BiasAdd/ReadVariableOp?
 sequential_104/dense_104/BiasAddBiasAdd)sequential_104/dense_104/MatMul:product:07sequential_104/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_104/dense_104/BiasAdd?
 sequential_104/dense_104/SigmoidSigmoid)sequential_104/dense_104/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_104/dense_104/Sigmoid?
Csequential_104/dense_104/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices?
1sequential_104/dense_104/ActivityRegularizer/MeanMean$sequential_104/dense_104/Sigmoid:y:0Lsequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_104/dense_104/ActivityRegularizer/Mean?
6sequential_104/dense_104/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_104/dense_104/ActivityRegularizer/Maximum/y?
4sequential_104/dense_104/ActivityRegularizer/MaximumMaximum:sequential_104/dense_104/ActivityRegularizer/Mean:output:0?sequential_104/dense_104/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_104/dense_104/ActivityRegularizer/Maximum?
6sequential_104/dense_104/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_104/dense_104/ActivityRegularizer/truediv/x?
4sequential_104/dense_104/ActivityRegularizer/truedivRealDiv?sequential_104/dense_104/ActivityRegularizer/truediv/x:output:08sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_104/dense_104/ActivityRegularizer/truediv?
0sequential_104/dense_104/ActivityRegularizer/LogLog8sequential_104/dense_104/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/Log?
2sequential_104/dense_104/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_104/dense_104/ActivityRegularizer/mul/x?
0sequential_104/dense_104/ActivityRegularizer/mulMul;sequential_104/dense_104/ActivityRegularizer/mul/x:output:04sequential_104/dense_104/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/mul?
2sequential_104/dense_104/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_104/dense_104/ActivityRegularizer/sub/x?
0sequential_104/dense_104/ActivityRegularizer/subSub;sequential_104/dense_104/ActivityRegularizer/sub/x:output:08sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/sub?
8sequential_104/dense_104/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_104/dense_104/ActivityRegularizer/truediv_1/x?
6sequential_104/dense_104/ActivityRegularizer/truediv_1RealDivAsequential_104/dense_104/ActivityRegularizer/truediv_1/x:output:04sequential_104/dense_104/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_104/dense_104/ActivityRegularizer/truediv_1?
2sequential_104/dense_104/ActivityRegularizer/Log_1Log:sequential_104/dense_104/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/Log_1?
4sequential_104/dense_104/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_104/dense_104/ActivityRegularizer/mul_1/x?
2sequential_104/dense_104/ActivityRegularizer/mul_1Mul=sequential_104/dense_104/ActivityRegularizer/mul_1/x:output:06sequential_104/dense_104/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/mul_1?
0sequential_104/dense_104/ActivityRegularizer/addAddV24sequential_104/dense_104/ActivityRegularizer/mul:z:06sequential_104/dense_104/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/add?
2sequential_104/dense_104/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_104/dense_104/ActivityRegularizer/Const?
0sequential_104/dense_104/ActivityRegularizer/SumSum4sequential_104/dense_104/ActivityRegularizer/add:z:0;sequential_104/dense_104/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/Sum?
4sequential_104/dense_104/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_104/dense_104/ActivityRegularizer/mul_2/x?
2sequential_104/dense_104/ActivityRegularizer/mul_2Mul=sequential_104/dense_104/ActivityRegularizer/mul_2/x:output:09sequential_104/dense_104/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/mul_2?
2sequential_104/dense_104/ActivityRegularizer/ShapeShape$sequential_104/dense_104/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_104/dense_104/ActivityRegularizer/Shape?
@sequential_104/dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_104/dense_104/ActivityRegularizer/strided_slice/stack?
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1?
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2?
:sequential_104/dense_104/ActivityRegularizer/strided_sliceStridedSlice;sequential_104/dense_104/ActivityRegularizer/Shape:output:0Isequential_104/dense_104/ActivityRegularizer/strided_slice/stack:output:0Ksequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_104/dense_104/ActivityRegularizer/strided_slice?
1sequential_104/dense_104/ActivityRegularizer/CastCastCsequential_104/dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_104/dense_104/ActivityRegularizer/Cast?
6sequential_104/dense_104/ActivityRegularizer/truediv_2RealDiv6sequential_104/dense_104/ActivityRegularizer/mul_2:z:05sequential_104/dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_104/dense_104/ActivityRegularizer/truediv_2?
.sequential_105/dense_105/MatMul/ReadVariableOpReadVariableOp7sequential_105_dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_105/dense_105/MatMul/ReadVariableOp?
sequential_105/dense_105/MatMulMatMul$sequential_104/dense_104/Sigmoid:y:06sequential_105/dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_105/dense_105/MatMul?
/sequential_105/dense_105/BiasAdd/ReadVariableOpReadVariableOp8sequential_105_dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_105/dense_105/BiasAdd/ReadVariableOp?
 sequential_105/dense_105/BiasAddBiasAdd)sequential_105/dense_105/MatMul:product:07sequential_105/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_105/dense_105/BiasAdd?
 sequential_105/dense_105/SigmoidSigmoid)sequential_105/dense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_105/dense_105/Sigmoid?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_104_dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_105_dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity$sequential_105/dense_105/Sigmoid:y:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp0^sequential_104/dense_104/BiasAdd/ReadVariableOp/^sequential_104/dense_104/MatMul/ReadVariableOp0^sequential_105/dense_105/BiasAdd/ReadVariableOp/^sequential_105/dense_105/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_104/dense_104/ActivityRegularizer/truediv_2:z:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp0^sequential_104/dense_104/BiasAdd/ReadVariableOp/^sequential_104/dense_104/MatMul/ReadVariableOp0^sequential_105/dense_105/BiasAdd/ReadVariableOp/^sequential_105/dense_105/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_104/dense_104/BiasAdd/ReadVariableOp/sequential_104/dense_104/BiasAdd/ReadVariableOp2`
.sequential_104/dense_104/MatMul/ReadVariableOp.sequential_104/dense_104/MatMul/ReadVariableOp2b
/sequential_105/dense_105/BiasAdd/ReadVariableOp/sequential_105/dense_105/BiasAdd/ReadVariableOp2`
.sequential_105/dense_105/MatMul/ReadVariableOp.sequential_105/dense_105/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_104_layer_call_fn_16640898
input_53
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_53unknown	unknown_0*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408802
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
input_53
?
?
1__inference_autoencoder_52_layer_call_fn_16641116
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_166411042
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640880

inputs$
dense_104_16640859:^  
dense_104_16640861: 
identity

identity_1??!dense_104/StatefulPartitionedCall?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinputsdense_104_16640859dense_104_16640861*
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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922#
!dense_104/StatefulPartitionedCall?
-dense_104/ActivityRegularizer/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
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
3__inference_dense_104_activity_regularizer_166407682/
-dense_104/ActivityRegularizer/PartitionedCall?
#dense_104/ActivityRegularizer/ShapeShape*dense_104/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
%dense_104/ActivityRegularizer/truedivRealDiv6dense_104/ActivityRegularizer/PartitionedCall:output:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_104_16640859*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_104/ActivityRegularizer/truediv:z:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?#
?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640946
input_53$
dense_104_16640925:^  
dense_104_16640927: 
identity

identity_1??!dense_104/StatefulPartitionedCall?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinput_53dense_104_16640925dense_104_16640927*
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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922#
!dense_104/StatefulPartitionedCall?
-dense_104/ActivityRegularizer/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
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
3__inference_dense_104_activity_regularizer_166407682/
-dense_104/ActivityRegularizer/PartitionedCall?
#dense_104/ActivityRegularizer/ShapeShape*dense_104/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
%dense_104/ActivityRegularizer/truedivRealDiv6dense_104/ActivityRegularizer/PartitionedCall:output:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_104_16640925*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_104/ActivityRegularizer/truediv:z:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_53
?%
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641242
input_1)
sequential_104_16641217:^ %
sequential_104_16641219: )
sequential_105_16641223: ^%
sequential_105_16641225:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?&sequential_104/StatefulPartitionedCall?&sequential_105/StatefulPartitionedCall?
&sequential_104/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_104_16641217sequential_104_16641219*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408802(
&sequential_104/StatefulPartitionedCall?
&sequential_105/StatefulPartitionedCallStatefulPartitionedCall/sequential_104/StatefulPartitionedCall:output:0sequential_105_16641223sequential_105_16641225*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166410262(
&sequential_105/StatefulPartitionedCall?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_104_16641217*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_105_16641223*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity/sequential_105/StatefulPartitionedCall:output:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_104/StatefulPartitionedCall:output:13^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_104/StatefulPartitionedCall&sequential_104/StatefulPartitionedCall2P
&sequential_105/StatefulPartitionedCall&sequential_105/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
S
3__inference_dense_104_activity_regularizer_16640768

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
&__inference_signature_wrapper_16641269
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
#__inference__wrapped_model_166407392
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
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16640983

inputs$
dense_105_16640971: ^ 
dense_105_16640973:^
identity??!dense_105/StatefulPartitionedCall?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallinputsdense_105_16640971dense_105_16640973*
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
G__inference_dense_105_layer_call_and_return_conditional_losses_166409702#
!dense_105/StatefulPartitionedCall?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_105_16640971*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641026

inputs$
dense_105_16641014: ^ 
dense_105_16641016:^
identity??!dense_105/StatefulPartitionedCall?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallinputsdense_105_16641014dense_105_16641016*
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
G__inference_dense_105_layer_call_and_return_conditional_losses_166409702#
!dense_105/StatefulPartitionedCall?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_105_16641014*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_dense_104_layer_call_fn_16641658

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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922
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
1__inference_sequential_105_layer_call_fn_16641557

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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166409832
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
?%
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641104
x)
sequential_104_16641079:^ %
sequential_104_16641081: )
sequential_105_16641085: ^%
sequential_105_16641087:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?&sequential_104/StatefulPartitionedCall?&sequential_105/StatefulPartitionedCall?
&sequential_104/StatefulPartitionedCallStatefulPartitionedCallxsequential_104_16641079sequential_104_16641081*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408142(
&sequential_104/StatefulPartitionedCall?
&sequential_105/StatefulPartitionedCallStatefulPartitionedCall/sequential_104/StatefulPartitionedCall:output:0sequential_105_16641085sequential_105_16641087*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166409832(
&sequential_105/StatefulPartitionedCall?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_104_16641079*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_105_16641085*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity/sequential_105/StatefulPartitionedCall:output:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_104/StatefulPartitionedCall:output:13^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_104/StatefulPartitionedCall&sequential_104/StatefulPartitionedCall2P
&sequential_105/StatefulPartitionedCall&sequential_105/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
!__inference__traced_save_16641775
file_prefix/
+savev2_dense_104_kernel_read_readvariableop-
)savev2_dense_104_bias_read_readvariableop/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_104_kernel_read_readvariableop)savev2_dense_104_bias_read_readvariableop+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?h
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641415
xI
7sequential_104_dense_104_matmul_readvariableop_resource:^ F
8sequential_104_dense_104_biasadd_readvariableop_resource: I
7sequential_105_dense_105_matmul_readvariableop_resource: ^F
8sequential_105_dense_105_biasadd_readvariableop_resource:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?/sequential_104/dense_104/BiasAdd/ReadVariableOp?.sequential_104/dense_104/MatMul/ReadVariableOp?/sequential_105/dense_105/BiasAdd/ReadVariableOp?.sequential_105/dense_105/MatMul/ReadVariableOp?
.sequential_104/dense_104/MatMul/ReadVariableOpReadVariableOp7sequential_104_dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_104/dense_104/MatMul/ReadVariableOp?
sequential_104/dense_104/MatMulMatMulx6sequential_104/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_104/dense_104/MatMul?
/sequential_104/dense_104/BiasAdd/ReadVariableOpReadVariableOp8sequential_104_dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_104/dense_104/BiasAdd/ReadVariableOp?
 sequential_104/dense_104/BiasAddBiasAdd)sequential_104/dense_104/MatMul:product:07sequential_104/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_104/dense_104/BiasAdd?
 sequential_104/dense_104/SigmoidSigmoid)sequential_104/dense_104/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_104/dense_104/Sigmoid?
Csequential_104/dense_104/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices?
1sequential_104/dense_104/ActivityRegularizer/MeanMean$sequential_104/dense_104/Sigmoid:y:0Lsequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_104/dense_104/ActivityRegularizer/Mean?
6sequential_104/dense_104/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_104/dense_104/ActivityRegularizer/Maximum/y?
4sequential_104/dense_104/ActivityRegularizer/MaximumMaximum:sequential_104/dense_104/ActivityRegularizer/Mean:output:0?sequential_104/dense_104/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_104/dense_104/ActivityRegularizer/Maximum?
6sequential_104/dense_104/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_104/dense_104/ActivityRegularizer/truediv/x?
4sequential_104/dense_104/ActivityRegularizer/truedivRealDiv?sequential_104/dense_104/ActivityRegularizer/truediv/x:output:08sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_104/dense_104/ActivityRegularizer/truediv?
0sequential_104/dense_104/ActivityRegularizer/LogLog8sequential_104/dense_104/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/Log?
2sequential_104/dense_104/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_104/dense_104/ActivityRegularizer/mul/x?
0sequential_104/dense_104/ActivityRegularizer/mulMul;sequential_104/dense_104/ActivityRegularizer/mul/x:output:04sequential_104/dense_104/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/mul?
2sequential_104/dense_104/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_104/dense_104/ActivityRegularizer/sub/x?
0sequential_104/dense_104/ActivityRegularizer/subSub;sequential_104/dense_104/ActivityRegularizer/sub/x:output:08sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/sub?
8sequential_104/dense_104/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_104/dense_104/ActivityRegularizer/truediv_1/x?
6sequential_104/dense_104/ActivityRegularizer/truediv_1RealDivAsequential_104/dense_104/ActivityRegularizer/truediv_1/x:output:04sequential_104/dense_104/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_104/dense_104/ActivityRegularizer/truediv_1?
2sequential_104/dense_104/ActivityRegularizer/Log_1Log:sequential_104/dense_104/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/Log_1?
4sequential_104/dense_104/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_104/dense_104/ActivityRegularizer/mul_1/x?
2sequential_104/dense_104/ActivityRegularizer/mul_1Mul=sequential_104/dense_104/ActivityRegularizer/mul_1/x:output:06sequential_104/dense_104/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/mul_1?
0sequential_104/dense_104/ActivityRegularizer/addAddV24sequential_104/dense_104/ActivityRegularizer/mul:z:06sequential_104/dense_104/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/add?
2sequential_104/dense_104/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_104/dense_104/ActivityRegularizer/Const?
0sequential_104/dense_104/ActivityRegularizer/SumSum4sequential_104/dense_104/ActivityRegularizer/add:z:0;sequential_104/dense_104/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_104/dense_104/ActivityRegularizer/Sum?
4sequential_104/dense_104/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_104/dense_104/ActivityRegularizer/mul_2/x?
2sequential_104/dense_104/ActivityRegularizer/mul_2Mul=sequential_104/dense_104/ActivityRegularizer/mul_2/x:output:09sequential_104/dense_104/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_104/dense_104/ActivityRegularizer/mul_2?
2sequential_104/dense_104/ActivityRegularizer/ShapeShape$sequential_104/dense_104/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_104/dense_104/ActivityRegularizer/Shape?
@sequential_104/dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_104/dense_104/ActivityRegularizer/strided_slice/stack?
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1?
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2?
:sequential_104/dense_104/ActivityRegularizer/strided_sliceStridedSlice;sequential_104/dense_104/ActivityRegularizer/Shape:output:0Isequential_104/dense_104/ActivityRegularizer/strided_slice/stack:output:0Ksequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_104/dense_104/ActivityRegularizer/strided_slice?
1sequential_104/dense_104/ActivityRegularizer/CastCastCsequential_104/dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_104/dense_104/ActivityRegularizer/Cast?
6sequential_104/dense_104/ActivityRegularizer/truediv_2RealDiv6sequential_104/dense_104/ActivityRegularizer/mul_2:z:05sequential_104/dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_104/dense_104/ActivityRegularizer/truediv_2?
.sequential_105/dense_105/MatMul/ReadVariableOpReadVariableOp7sequential_105_dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_105/dense_105/MatMul/ReadVariableOp?
sequential_105/dense_105/MatMulMatMul$sequential_104/dense_104/Sigmoid:y:06sequential_105/dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_105/dense_105/MatMul?
/sequential_105/dense_105/BiasAdd/ReadVariableOpReadVariableOp8sequential_105_dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_105/dense_105/BiasAdd/ReadVariableOp?
 sequential_105/dense_105/BiasAddBiasAdd)sequential_105/dense_105/MatMul:product:07sequential_105/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_105/dense_105/BiasAdd?
 sequential_105/dense_105/SigmoidSigmoid)sequential_105/dense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_105/dense_105/Sigmoid?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_104_dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_105_dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity$sequential_105/dense_105/Sigmoid:y:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp0^sequential_104/dense_104/BiasAdd/ReadVariableOp/^sequential_104/dense_104/MatMul/ReadVariableOp0^sequential_105/dense_105/BiasAdd/ReadVariableOp/^sequential_105/dense_105/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_104/dense_104/ActivityRegularizer/truediv_2:z:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp0^sequential_104/dense_104/BiasAdd/ReadVariableOp/^sequential_104/dense_104/MatMul/ReadVariableOp0^sequential_105/dense_105/BiasAdd/ReadVariableOp/^sequential_105/dense_105/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_104/dense_104/BiasAdd/ReadVariableOp/sequential_104/dense_104/BiasAdd/ReadVariableOp2`
.sequential_104/dense_104/MatMul/ReadVariableOp.sequential_104/dense_104/MatMul/ReadVariableOp2b
/sequential_105/dense_105/BiasAdd/ReadVariableOp/sequential_105/dense_105/BiasAdd/ReadVariableOp2`
.sequential_105/dense_105/MatMul/ReadVariableOp.sequential_105/dense_105/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_104_layer_call_fn_16641431

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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408142
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
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641609

inputs:
(dense_105_matmul_readvariableop_resource: ^7
)dense_105_biasadd_readvariableop_resource:^
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulinputs'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/BiasAdd
dense_105/SigmoidSigmoiddense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_105/Sigmoid?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitydense_105/Sigmoid:y:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_105_layer_call_fn_16641548
dense_105_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_105_inputunknown	unknown_0*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166409832
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
_user_specified_namedense_105_input
?
?
$__inference__traced_restore_16641797
file_prefix3
!assignvariableop_dense_104_kernel:^ /
!assignvariableop_1_dense_104_bias: 5
#assignvariableop_2_dense_105_kernel: ^/
!assignvariableop_3_dense_105_bias:^

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
AssignVariableOpAssignVariableOp!assignvariableop_dense_104_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_104_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_105_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_105_biasIdentity_3:output:0"/device:CPU:0*
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
?
?
1__inference_sequential_104_layer_call_fn_16640822
input_53
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_53unknown	unknown_0*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408142
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
input_53
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641592

inputs:
(dense_105_matmul_readvariableop_resource: ^7
)dense_105_biasadd_readvariableop_resource:^
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulinputs'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/BiasAdd
dense_105/SigmoidSigmoiddense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_105/Sigmoid?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitydense_105/Sigmoid:y:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_104_layer_call_fn_16641441

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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408802
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
__inference_loss_fn_1_16641723M
;dense_105_kernel_regularizer_square_readvariableop_resource: ^
identity??2dense_105/kernel/Regularizer/Square/ReadVariableOp?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_105_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity$dense_105/kernel/Regularizer/mul:z:03^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp
?
?
1__inference_autoencoder_52_layer_call_fn_16641297
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_166411602
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
?#
?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640814

inputs$
dense_104_16640793:^  
dense_104_16640795: 
identity

identity_1??!dense_104/StatefulPartitionedCall?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinputsdense_104_16640793dense_104_16640795*
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
G__inference_dense_104_layer_call_and_return_conditional_losses_166407922#
!dense_104/StatefulPartitionedCall?
-dense_104/ActivityRegularizer/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
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
3__inference_dense_104_activity_regularizer_166407682/
-dense_104/ActivityRegularizer/PartitionedCall?
#dense_104/ActivityRegularizer/ShapeShape*dense_104/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
%dense_104/ActivityRegularizer/truedivRealDiv6dense_104/ActivityRegularizer/PartitionedCall:output:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_104_16640793*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_104/ActivityRegularizer/truediv:z:0"^dense_104/StatefulPartitionedCall3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
,__inference_dense_105_layer_call_fn_16641695

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
G__inference_dense_105_layer_call_and_return_conditional_losses_166409702
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
?_
?
#__inference__wrapped_model_16640739
input_1X
Fautoencoder_52_sequential_104_dense_104_matmul_readvariableop_resource:^ U
Gautoencoder_52_sequential_104_dense_104_biasadd_readvariableop_resource: X
Fautoencoder_52_sequential_105_dense_105_matmul_readvariableop_resource: ^U
Gautoencoder_52_sequential_105_dense_105_biasadd_readvariableop_resource:^
identity??>autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp?=autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp?>autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp?=autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp?
=autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOpReadVariableOpFautoencoder_52_sequential_104_dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02?
=autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp?
.autoencoder_52/sequential_104/dense_104/MatMulMatMulinput_1Eautoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.autoencoder_52/sequential_104/dense_104/MatMul?
>autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_52_sequential_104_dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp?
/autoencoder_52/sequential_104/dense_104/BiasAddBiasAdd8autoencoder_52/sequential_104/dense_104/MatMul:product:0Fautoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_52/sequential_104/dense_104/BiasAdd?
/autoencoder_52/sequential_104/dense_104/SigmoidSigmoid8autoencoder_52/sequential_104/dense_104/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_52/sequential_104/dense_104/Sigmoid?
Rautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices?
@autoencoder_52/sequential_104/dense_104/ActivityRegularizer/MeanMean3autoencoder_52/sequential_104/dense_104/Sigmoid:y:0[autoencoder_52/sequential_104/dense_104/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2B
@autoencoder_52/sequential_104/dense_104/ActivityRegularizer/Mean?
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2G
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum/y?
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/MaximumMaximumIautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Mean:output:0Nautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2E
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum?
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2G
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv/x?
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truedivRealDivNautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv/x:output:0Gautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv?
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/LogLogGautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2A
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/Log?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul/x?
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/mulMulJautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul/x:output:0Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2A
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/sub/x?
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/subSubJautoencoder_52/sequential_104/dense_104/ActivityRegularizer/sub/x:output:0Gautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2A
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/sub?
Gautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2I
Gautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1/x?
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1RealDivPautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1/x:output:0Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2G
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Log_1LogIautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Log_1?
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2E
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1/x?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1MulLautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1/x:output:0Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1?
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/addAddV2Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul:z:0Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/add?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Const?
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/SumSumCautoencoder_52/sequential_104/dense_104/ActivityRegularizer/add:z:0Jautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2A
?autoencoder_52/sequential_104/dense_104/ActivityRegularizer/Sum?
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
Cautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2/x?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2MulLautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2/x:output:0Hautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2?
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/ShapeShape3autoencoder_52/sequential_104/dense_104/Sigmoid:y:0*
T0*
_output_shapes
:2C
Aautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Shape?
Oautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack?
Qautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1?
Qautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2?
Iautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_sliceStridedSliceJautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Shape:output:0Xautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack:output:0Zautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_1:output:0Zautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice?
@autoencoder_52/sequential_104/dense_104/ActivityRegularizer/CastCastRautoencoder_52/sequential_104/dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@autoencoder_52/sequential_104/dense_104/ActivityRegularizer/Cast?
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_2RealDivEautoencoder_52/sequential_104/dense_104/ActivityRegularizer/mul_2:z:0Dautoencoder_52/sequential_104/dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2G
Eautoencoder_52/sequential_104/dense_104/ActivityRegularizer/truediv_2?
=autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOpReadVariableOpFautoencoder_52_sequential_105_dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02?
=autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp?
.autoencoder_52/sequential_105/dense_105/MatMulMatMul3autoencoder_52/sequential_104/dense_104/Sigmoid:y:0Eautoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^20
.autoencoder_52/sequential_105/dense_105/MatMul?
>autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_52_sequential_105_dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02@
>autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp?
/autoencoder_52/sequential_105/dense_105/BiasAddBiasAdd8autoencoder_52/sequential_105/dense_105/MatMul:product:0Fautoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_52/sequential_105/dense_105/BiasAdd?
/autoencoder_52/sequential_105/dense_105/SigmoidSigmoid8autoencoder_52/sequential_105/dense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_52/sequential_105/dense_105/Sigmoid?
IdentityIdentity3autoencoder_52/sequential_105/dense_105/Sigmoid:y:0?^autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp>^autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp?^autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp>^autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2?
>autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp>autoencoder_52/sequential_104/dense_104/BiasAdd/ReadVariableOp2~
=autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp=autoencoder_52/sequential_104/dense_104/MatMul/ReadVariableOp2?
>autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp>autoencoder_52/sequential_105/dense_105/BiasAdd/ReadVariableOp2~
=autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp=autoencoder_52/sequential_105/dense_105/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?%
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641160
x)
sequential_104_16641135:^ %
sequential_104_16641137: )
sequential_105_16641141: ^%
sequential_105_16641143:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?&sequential_104/StatefulPartitionedCall?&sequential_105/StatefulPartitionedCall?
&sequential_104/StatefulPartitionedCallStatefulPartitionedCallxsequential_104_16641135sequential_104_16641137*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408802(
&sequential_104/StatefulPartitionedCall?
&sequential_105/StatefulPartitionedCallStatefulPartitionedCall/sequential_104/StatefulPartitionedCall:output:0sequential_105_16641141sequential_105_16641143*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166410262(
&sequential_105/StatefulPartitionedCall?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_104_16641135*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_105_16641141*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity/sequential_105/StatefulPartitionedCall:output:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_104/StatefulPartitionedCall:output:13^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_104/StatefulPartitionedCall&sequential_104/StatefulPartitionedCall2P
&sequential_105/StatefulPartitionedCall&sequential_105/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?%
?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641214
input_1)
sequential_104_16641189:^ %
sequential_104_16641191: )
sequential_105_16641195: ^%
sequential_105_16641197:^
identity

identity_1??2dense_104/kernel/Regularizer/Square/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?&sequential_104/StatefulPartitionedCall?&sequential_105/StatefulPartitionedCall?
&sequential_104/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_104_16641189sequential_104_16641191*
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_166408142(
&sequential_104/StatefulPartitionedCall?
&sequential_105/StatefulPartitionedCallStatefulPartitionedCall/sequential_104/StatefulPartitionedCall:output:0sequential_105_16641195sequential_105_16641197*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166409832(
&sequential_105/StatefulPartitionedCall?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_104_16641189*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_105_16641195*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentity/sequential_105/StatefulPartitionedCall:output:03^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_104/StatefulPartitionedCall:output:13^dense_104/kernel/Regularizer/Square/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp'^sequential_104/StatefulPartitionedCall'^sequential_105/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_104/StatefulPartitionedCall&sequential_104/StatefulPartitionedCall2P
&sequential_105/StatefulPartitionedCall&sequential_105/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_autoencoder_52_layer_call_fn_16641186
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_166411602
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
?B
?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641533

inputs:
(dense_104_matmul_readvariableop_resource:^ 7
)dense_104_biasadd_readvariableop_resource: 
identity

identity_1?? dense_104/BiasAdd/ReadVariableOp?dense_104/MatMul/ReadVariableOp?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_104/MatMul/ReadVariableOp?
dense_104/MatMulMatMulinputs'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_104/MatMul?
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_104/BiasAdd/ReadVariableOp?
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_104/BiasAdd
dense_104/SigmoidSigmoiddense_104/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_104/Sigmoid?
4dense_104/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_104/ActivityRegularizer/Mean/reduction_indices?
"dense_104/ActivityRegularizer/MeanMeandense_104/Sigmoid:y:0=dense_104/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Mean?
'dense_104/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_104/ActivityRegularizer/Maximum/y?
%dense_104/ActivityRegularizer/MaximumMaximum+dense_104/ActivityRegularizer/Mean:output:00dense_104/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/Maximum?
'dense_104/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_104/ActivityRegularizer/truediv/x?
%dense_104/ActivityRegularizer/truedivRealDiv0dense_104/ActivityRegularizer/truediv/x:output:0)dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_104/ActivityRegularizer/truediv?
!dense_104/ActivityRegularizer/LogLog)dense_104/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/Log?
#dense_104/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_104/ActivityRegularizer/mul/x?
!dense_104/ActivityRegularizer/mulMul,dense_104/ActivityRegularizer/mul/x:output:0%dense_104/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/mul?
#dense_104/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_104/ActivityRegularizer/sub/x?
!dense_104/ActivityRegularizer/subSub,dense_104/ActivityRegularizer/sub/x:output:0)dense_104/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/sub?
)dense_104/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_104/ActivityRegularizer/truediv_1/x?
'dense_104/ActivityRegularizer/truediv_1RealDiv2dense_104/ActivityRegularizer/truediv_1/x:output:0%dense_104/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_104/ActivityRegularizer/truediv_1?
#dense_104/ActivityRegularizer/Log_1Log+dense_104/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/Log_1?
%dense_104/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_104/ActivityRegularizer/mul_1/x?
#dense_104/ActivityRegularizer/mul_1Mul.dense_104/ActivityRegularizer/mul_1/x:output:0'dense_104/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/mul_1?
!dense_104/ActivityRegularizer/addAddV2%dense_104/ActivityRegularizer/mul:z:0'dense_104/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/add?
#dense_104/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_104/ActivityRegularizer/Const?
!dense_104/ActivityRegularizer/SumSum%dense_104/ActivityRegularizer/add:z:0,dense_104/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_104/ActivityRegularizer/Sum?
%dense_104/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_104/ActivityRegularizer/mul_2/x?
#dense_104/ActivityRegularizer/mul_2Mul.dense_104/ActivityRegularizer/mul_2/x:output:0*dense_104/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_104/ActivityRegularizer/mul_2?
#dense_104/ActivityRegularizer/ShapeShapedense_104/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_104/ActivityRegularizer/Shape?
1dense_104/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_104/ActivityRegularizer/strided_slice/stack?
3dense_104/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_1?
3dense_104/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_104/ActivityRegularizer/strided_slice/stack_2?
+dense_104/ActivityRegularizer/strided_sliceStridedSlice,dense_104/ActivityRegularizer/Shape:output:0:dense_104/ActivityRegularizer/strided_slice/stack:output:0<dense_104/ActivityRegularizer/strided_slice/stack_1:output:0<dense_104/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_104/ActivityRegularizer/strided_slice?
"dense_104/ActivityRegularizer/CastCast4dense_104/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_104/ActivityRegularizer/Cast?
'dense_104/ActivityRegularizer/truediv_2RealDiv'dense_104/ActivityRegularizer/mul_2:z:0&dense_104/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_104/ActivityRegularizer/truediv_2?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentitydense_104/Sigmoid:y:0!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_104/ActivityRegularizer/truediv_2:z:0!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641626
dense_105_input:
(dense_105_matmul_readvariableop_resource: ^7
)dense_105_biasadd_readvariableop_resource:^
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMuldense_105_input'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_105/BiasAdd
dense_105/SigmoidSigmoiddense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_105/Sigmoid?
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitydense_105/Sigmoid:y:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2h
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_105_input
?
?
G__inference_dense_104_layer_call_and_return_conditional_losses_16641740

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_16641680M
;dense_104_kernel_regularizer_square_readvariableop_resource:^ 
identity??2dense_104/kernel/Regularizer/Square/ReadVariableOp?
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_104_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentity$dense_104/kernel/Regularizer/mul:z:03^dense_104/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp
?
?
G__inference_dense_105_layer_call_and_return_conditional_losses_16641712

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_105/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_105/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_105/kernel/Regularizer/Square/ReadVariableOp?
#dense_105/kernel/Regularizer/SquareSquare:dense_105/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_105/kernel/Regularizer/Square?
"dense_105/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_105/kernel/Regularizer/Const?
 dense_105/kernel/Regularizer/SumSum'dense_105/kernel/Regularizer/Square:y:0+dense_105/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/Sum?
"dense_105/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_105/kernel/Regularizer/mul/x?
 dense_105/kernel/Regularizer/mulMul+dense_105/kernel/Regularizer/mul/x:output:0)dense_105/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_105/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_105/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_105/kernel/Regularizer/Square/ReadVariableOp2dense_105/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_104_layer_call_and_return_conditional_losses_16640792

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_104/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_104/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_104/kernel/Regularizer/Square/ReadVariableOp?
#dense_104/kernel/Regularizer/SquareSquare:dense_104/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_104/kernel/Regularizer/Square?
"dense_104/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_104/kernel/Regularizer/Const?
 dense_104/kernel/Regularizer/SumSum'dense_104/kernel/Regularizer/Square:y:0+dense_104/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/Sum?
"dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_104/kernel/Regularizer/mul/x?
 dense_104/kernel/Regularizer/mulMul+dense_104/kernel/Regularizer/mul/x:output:0)dense_104/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_104/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_104/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_104/kernel/Regularizer/Square/ReadVariableOp2dense_104/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_105_layer_call_fn_16641575
dense_105_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_105_inputunknown	unknown_0*
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_166410262
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
_user_specified_namedense_105_input"?L
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
_tf_keras_model?{"name": "autoencoder_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
_tf_keras_sequential?{"name": "sequential_104", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_104", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}}, {"class_name": "Dense", "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_53"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_104", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_105", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_105", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_105_input"}}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_105_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_105", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_105_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
{"name": "dense_104", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
_tf_keras_layer?{"name": "dense_105", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
": ^ 2dense_104/kernel
: 2dense_104/bias
":  ^2dense_105/kernel
:^2dense_105/bias
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
1__inference_autoencoder_52_layer_call_fn_16641116
1__inference_autoencoder_52_layer_call_fn_16641283
1__inference_autoencoder_52_layer_call_fn_16641297
1__inference_autoencoder_52_layer_call_fn_16641186?
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
#__inference__wrapped_model_16640739?
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641356
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641415
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641214
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641242?
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
1__inference_sequential_104_layer_call_fn_16640822
1__inference_sequential_104_layer_call_fn_16641431
1__inference_sequential_104_layer_call_fn_16641441
1__inference_sequential_104_layer_call_fn_16640898?
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641487
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641533
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640922
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640946?
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
1__inference_sequential_105_layer_call_fn_16641548
1__inference_sequential_105_layer_call_fn_16641557
1__inference_sequential_105_layer_call_fn_16641566
1__inference_sequential_105_layer_call_fn_16641575?
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641592
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641609
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641626
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641643?
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
&__inference_signature_wrapper_16641269input_1"?
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
,__inference_dense_104_layer_call_fn_16641658?
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
K__inference_dense_104_layer_call_and_return_all_conditional_losses_16641669?
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
__inference_loss_fn_0_16641680?
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
,__inference_dense_105_layer_call_fn_16641695?
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
G__inference_dense_105_layer_call_and_return_conditional_losses_16641712?
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
__inference_loss_fn_1_16641723?
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
3__inference_dense_104_activity_regularizer_16640768?
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
G__inference_dense_104_layer_call_and_return_conditional_losses_16641740?
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
#__inference__wrapped_model_16640739m0?-
&?#
!?
input_1?????????^
? "3?0
.
output_1"?
output_1?????????^?
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641214q4?1
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641242q4?1
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641356k.?+
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
L__inference_autoencoder_52_layer_call_and_return_conditional_losses_16641415k.?+
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
1__inference_autoencoder_52_layer_call_fn_16641116V4?1
*?'
!?
input_1?????????^
p 
? "??????????^?
1__inference_autoencoder_52_layer_call_fn_16641186V4?1
*?'
!?
input_1?????????^
p
? "??????????^?
1__inference_autoencoder_52_layer_call_fn_16641283P.?+
$?!
?
X?????????^
p 
? "??????????^?
1__inference_autoencoder_52_layer_call_fn_16641297P.?+
$?!
?
X?????????^
p
? "??????????^f
3__inference_dense_104_activity_regularizer_16640768/$?!
?
?

activation
? "? ?
K__inference_dense_104_layer_call_and_return_all_conditional_losses_16641669j/?,
%?"
 ?
inputs?????????^
? "3?0
?
0????????? 
?
?	
1/0 ?
G__inference_dense_104_layer_call_and_return_conditional_losses_16641740\/?,
%?"
 ?
inputs?????????^
? "%?"
?
0????????? 
? 
,__inference_dense_104_layer_call_fn_16641658O/?,
%?"
 ?
inputs?????????^
? "?????????? ?
G__inference_dense_105_layer_call_and_return_conditional_losses_16641712\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????^
? 
,__inference_dense_105_layer_call_fn_16641695O/?,
%?"
 ?
inputs????????? 
? "??????????^=
__inference_loss_fn_0_16641680?

? 
? "? =
__inference_loss_fn_1_16641723?

? 
? "? ?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640922t9?6
/?,
"?
input_53?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16640946t9?6
/?,
"?
input_53?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641487r7?4
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
L__inference_sequential_104_layer_call_and_return_conditional_losses_16641533r7?4
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
1__inference_sequential_104_layer_call_fn_16640822Y9?6
/?,
"?
input_53?????????^
p 

 
? "?????????? ?
1__inference_sequential_104_layer_call_fn_16640898Y9?6
/?,
"?
input_53?????????^
p

 
? "?????????? ?
1__inference_sequential_104_layer_call_fn_16641431W7?4
-?*
 ?
inputs?????????^
p 

 
? "?????????? ?
1__inference_sequential_104_layer_call_fn_16641441W7?4
-?*
 ?
inputs?????????^
p

 
? "?????????? ?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641592d7?4
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641609d7?4
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
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641626m@?=
6?3
)?&
dense_105_input????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_105_layer_call_and_return_conditional_losses_16641643m@?=
6?3
)?&
dense_105_input????????? 
p

 
? "%?"
?
0?????????^
? ?
1__inference_sequential_105_layer_call_fn_16641548`@?=
6?3
)?&
dense_105_input????????? 
p 

 
? "??????????^?
1__inference_sequential_105_layer_call_fn_16641557W7?4
-?*
 ?
inputs????????? 
p 

 
? "??????????^?
1__inference_sequential_105_layer_call_fn_16641566W7?4
-?*
 ?
inputs????????? 
p

 
? "??????????^?
1__inference_sequential_105_layer_call_fn_16641575`@?=
6?3
)?&
dense_105_input????????? 
p

 
? "??????????^?
&__inference_signature_wrapper_16641269x;?8
? 
1?.
,
input_1!?
input_1?????????^"3?0
.
output_1"?
output_1?????????^