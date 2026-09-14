const { Document, Packer, Paragraph, TextRun, HeadingLevel, LevelFormat, AlignmentType } = require('docx');

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "highlights-bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 } }, // US Letter
      },
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [
            new TextRun({
              text: "Highlights",
              bold: true,
            }),
          ],
        }),
        new Paragraph({
          spacing: { after: 200 },
          children: [
            new TextRun({
              text: "Quantifying Subgroup-Valid Uncertainty in Superconducting Critical Temperature Prediction: A Locally-Normalized Mondrian Conformal Prediction Framework",
              italics: true,
            }),
          ],
        }),
        new Paragraph({
          spacing: { after: 100 },
          children: [new TextRun({ text: "Mahamudul Hassan Siddique" })],
        }),
        new Paragraph({
          spacing: { after: 300 },
          children: [
            new TextRun({
              text: "Department of Industrial and Production Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh",
              size: 20,
            }),
          ],
        }),

        new Paragraph({
          numbering: { reference: "highlights-bullets", level: 0 },
          spacing: { after: 160 },
          children: [
            new TextRun({
              text: "Six widely used machine learning algorithms (random forest, extra trees, gradient boosting, k-nearest neighbors, ridge regression, and support vector regression) are benchmarked for predicting superconducting critical temperature, and the best-performing model, extra trees, is selected via Bayesian hyperparameter optimization.",
            }),
          ],
        }),
        new Paragraph({
          numbering: { reference: "highlights-bullets", level: 0 },
          spacing: { after: 160 },
          children: [
            new TextRun({
              text: "Marginal split conformal prediction attains valid average coverage but undercovers simple, low-element compositions by more than ten percentage points below the nominal target.",
            }),
          ],
        }),
        new Paragraph({
          numbering: { reference: "highlights-bullets", level: 0 },
          spacing: { after: 160 },
          children: [
            new TextRun({
              text: "A locally-normalized Mondrian conformal prediction framework, combining ensemble-spread normalization with group-conditional calibration, restores per-group validity while narrowing intervals by up to 23% relative to standard calibration.",
            }),
          ],
        }),
        new Paragraph({
          numbering: { reference: "highlights-bullets", level: 0 },
          spacing: { after: 160 },
          children: [
            new TextRun({
              text: "Bayesian hyperparameter optimization confirms that the residual train-validation gap is not attributable to correctable overfitting, and that tuning's main practical benefit is computational efficiency rather than accuracy.",
            }),
          ],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  require('fs').writeFileSync('/home/claude/paper/highlights.docx', buffer);
  console.log('written');
});
