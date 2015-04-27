package com.garg.neuralnetwork

import java.util.Random

object AndTestApp extends App {
  val output: List[Double] = List(0, 0.0, 0.0, 1.0)

  val input: List[List[Double]] =
    List(List(0, 0), List(0, 1), List(1, 0), List(1, 1)).map(x => List(1.0) ++ x.map(_.toDouble))

  val weights = input.head.map(x => rand).toArray

  for (i <- 0 until 100000 + 1) {

    val y_hat = input.map(i => sigmoid(i.zip(weights).map(e => e._1 * e._2).sum))
    val errors = output.zip(y_hat).map(t => t._1 - t._2).toList
    val residualDerivatives = input.zipWithIndex.map(ir => {
      val row = ir._2
      ir._1.map(Xi => -1.0 * errors(row) * y_hat(row) * (1.0 - y_hat(row)) * Xi)
    }).transpose

    weights.zipWithIndex.zip(residualDerivatives.map(_.sum)).foreach {
      wid =>
        val Wold = wid._1._1
        val index = wid._1._2
        val delta = wid._2

        weights(index) = Wold - (1.0 / input.size) * (0.1 * delta)

    }
    if (i % 1000 == 0) {
      //println(residualDerivatives.mkString("\n"))
      // println("Weights: " + weights.mkString(","))
      println("Outputs: " + y_hat.mkString(","))
      //   println("Expected: " + output.mkString(","))
      // println("Errors " + errors.map(math.abs(_)))
      println("Errors Sum " + errors.map(math.abs(_)).sum)
    }
  }

  private def rand = (math.random * .4 - .2)
  private def sigmoid(x: Double) = 1.0 / (1.0 + math.exp(-x))
}