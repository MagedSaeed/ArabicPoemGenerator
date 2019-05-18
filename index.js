$(document).ready(function() {
  idx2char = [
    "\t",
    "\n",
    " ",
    "ء",
    "آ",
    "أ",
    "ؤ",
    "إ",
    "ئ",
    "ا",
    "ب",
    "ة",
    "ت",
    "ث",
    "ج",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ك",
    "ل",
    "م",
    "ن",
    "ه",
    "و",
    "ى",
    "ي"
  ];

  char2idx = {
    "\t": 0,
    "\n": 1,
    " ": 2,
    ء: 3,
    آ: 4,
    أ: 5,
    ؤ: 6,
    إ: 7,
    ئ: 8,
    ا: 9,
    ب: 10,
    ة: 11,
    ت: 12,
    ث: 13,
    ج: 14,
    ح: 15,
    خ: 16,
    د: 17,
    ذ: 18,
    ر: 19,
    ز: 20,
    س: 21,
    ش: 22,
    ص: 23,
    ض: 24,
    ط: 25,
    ظ: 26,
    ع: 27,
    غ: 28,
    ف: 29,
    ق: 30,
    ك: 31,
    ل: 32,
    م: 33,
    ن: 34,
    ه: 35,
    و: 36,
    ى: 37,
    ي: 38
  };
  model = null;
  $(".inputs").hide();
  $(".outputs").hide();
  start();
  async function start() {
    //load the model
    model = await tf.loadLayersModel("model-200/model.json");
    // generate_text(model);
    $("#loader").hide();
    $(".inputs").show("slow");
    $(".outputs").show("slow");
  }

  $(".submit").on("click", function() {
    generate_text(model);
  });

  async function generate_text(model) {
    // start_string = "سلام";
    model.summary();
    start_string = $(".inputs input").val();
    console.log(start_string);

    num_generate = 250;

    input_eval = [];

    for (var i = 0; i < start_string.length; i++) {
      input_eval.push(char2idx[start_string.charAt(i)]);
    }

    input_eval = tf.tensor(input_eval);
    input_eval = tf.expandDims(input_eval, 0);

    text_generated = [];

    temperature = 1.0;
    console.log("looping");
    $(".output-text").text(start_string);
    new_line_counter = 0;
    while (new_line_counter < 10) {
      predictions = model.predict(input_eval);
      predictions = tf.squeeze(predictions, 0);

      predicted_id = tf.multinomial(predictions, (num_samples = 1));
      predicted_id = predicted_id.dataSync()[0];
      input_eval = tf.expandDims([predicted_id], 0);
      next_char = idx2char[predicted_id];
      text_generated.push(idx2char[predicted_id]);
      if (next_char === "\t") continue;
      $(".output-text").text($(".output-text").text() + next_char);
      await tf.nextFrame();
      if (next_char === "\n") new_line_counter++;
    }
    for (i = 0; i < num_generate; i++) {
      predictions = model.predict(input_eval);
      predictions = tf.squeeze(predictions, 0);

      predicted_id = tf.multinomial(predictions, (num_samples = 1));
      predicted_id = predicted_id.dataSync()[0];
      input_eval = tf.expandDims([predicted_id], 0);
      next_char = idx2char[predicted_id];
      text_generated.push(idx2char[predicted_id]);
      if (next_char === "\t") continue;
      $(".output-text").text($(".output-text").text() + next_char);
      await tf.nextFrame();
    }

    console.log(text_generated.join(""));
    // return text_generated;
  }
});
