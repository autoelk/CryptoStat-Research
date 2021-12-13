package edu.rit.ctrl;

import edu.rit.crst.Function;
import edu.rit.gpu.Gpu;
import edu.rit.util.Random;
import edu.rit.util.BigInt;
import java.io.IOException;

public class Alternate extends Function {

  private int[] W = new int [16];
 
  public Alternate() {
    super();
  }

  public String constructor() {
    return "edu.rit.ctrl.Alternate()";
  }

  public String description() {
    return "a function that produces alternating rounds of zeroes and random sequences";
  }

  // input A
  public String A_description() {
    return "plaintext";
  }

  public int A_bitSize() {
    return 128;
  }

  // input B
  public String B_description() {
    return "key";
  }

  public int B_bitSize() {
    return 128;
  }

  // output C
  public String C_description() {
    return "ciphertext";
  }

  public int C_bitSize() {
    return 128;
  }

  // number of rounds
  public int rounds() {
    return 10;
  }

  // implementation for function
  public void evaluate(BigInt A, BigInt B, BigInt[] C) {
  }

  protected String moduleName() {
    return "edu/rit/ctrl/Alternate.ptx";
  }
}