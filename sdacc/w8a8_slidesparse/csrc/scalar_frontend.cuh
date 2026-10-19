// Matched original scalar-load C++ control; no second GPU quantization kernel.
at::Tensor projection_scalar(at::Tensor const& input,at::Tensor const& packed,
    at::Tensor const& metadata,at::Tensor const& scales,at::Tensor const& bias,
    int64_t group,int64_t n,int64_t config,c10::optional<at::Tensor> const& addend) {
  TORCH_CHECK(group==4,"scalar control supports2:4 only");
  auto result=quantize_scalar(input);
  int64_t m=input.numel()/input.size(-1);
  auto y=sdacc_dualscale_sm80::run(std::get<0>(result),packed,metadata,
    std::get<1>(result),scales,bias,m,n,config,addend);
  auto shape=input.sizes().vec();shape.back()=n;return y.reshape(shape);
}
