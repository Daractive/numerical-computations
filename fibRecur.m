function f=fibRecur(n)
  if n <= 2
    f=1.0;
  else
    f=fibRecur(n-1)+fibRecur(n-2);
  end
end
