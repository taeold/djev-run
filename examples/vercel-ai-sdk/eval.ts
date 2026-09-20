import {createTypeSafeAi} from '@ai-sdk/typesafe-ai';
import {experimental_evaluate, type Experimental_EvaluationModel} from 'ai';

const typeSafeAi = createTypeSafeAi({
  baseURL: process.env.DJEV_BASE_URL ?? 'https://<your-cloud-run-url>/v1',
  apiKey: process.env.DJEV_TOKEN,
});

async function triage(model: Experimental_EvaluationModel, message: string) {
  return experimental_evaluate({
    model,
    state: {message},
    questions: {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {
          billing: 'Payments and refunds',
          support: 'Other requests',
        },
      },
      severity: {
        type: 'score',
        instructions: 'How severe is the issue?',
        criteria: ['Cosmetic', 'Workaround exists', 'Blocking; no workaround'],
      },
      requestsRefund: {
        type: 'boolean',
        instructions: 'Is the customer requesting money back?',
      },
    },
  });
}

const result = await triage(
  typeSafeAi.evaluationModel('jev-latest'),
  'I was charged twice and my account is locked',
);

console.log(result.answers);
