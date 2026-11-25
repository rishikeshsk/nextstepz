import type {
  ExpertiseQuestion,
  ExpertiseQuestionOutcome,
  ExpertiseQuiz,
  ExpertiseTestResult,
} from '../types/internship';

export type ExpertiseArea =
  | 'Cloud & DevOps'
  | 'Data Science & AI/ML'
  | 'Full-Stack Web Engineering'
  | 'Cybersecurity & Networking'
  | 'Embedded Systems & IoT'
  | 'Core Software & Databases';

interface DomainConfig {
  area: ExpertiseArea;
  keywords: string[];
  questions: ExpertiseQuestion[];
}

function normalizeCorpus(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9+#.\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function hasWordBoundaryMatch(corpus: string, keyword: string): boolean {
  const norm = normalizeCorpus(keyword);
  if (!norm) return false;
  const escaped = norm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?:^|\\s)${escaped}(?:$|\\s)`).test(corpus);
}

const DOMAIN_BANK: DomainConfig[] = [
  {
    area: 'Cloud & DevOps',
    keywords: [
      'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'devops',
      'ci/cd', 'jenkins', 'linux', 'ec2', 's3', 'cloud', 'ansible', 'nginx',
    ],
    questions: [
      {
        id: 'cloud-1',
        topic: 'Containers & Docker',
        prompt:
          'When writing a production Dockerfile for a Python or Node service, what is the primary benefit of using a multi-stage build?',
        options: [
          'It lets a single container run two separate kernels simultaneously.',
          'It leaves build compilers and dev dependencies out of the final runtime image, shrinking size and attack surface.',
          'It automatically deploys the image across multiple AWS availability zones.',
          'It disables layer caching so every rebuild installs packages from scratch.',
        ],
        correctIndex: 1,
        explanation:
          'Multi-stage builds copy only compiled artifacts or runtime packages into a slim final image, reducing image size and vulnerabilities.',
      },
      {
        id: 'cloud-2',
        topic: 'Kubernetes Architecture',
        prompt:
          'In Kubernetes, what resource ensures a stable DNS name and load-balanced endpoint even as individual Pods restart with new IP addresses?',
        options: [
          'A ConfigMap',
          'A Service',
          'A PersistentVolumeClaim',
          'A DaemonSet',
        ],
        correctIndex: 1,
        explanation:
          'A Kubernetes Service maps a stable virtual IP and DNS name to dynamic Pods matched by label selectors.',
      },
      {
        id: 'cloud-3',
        topic: 'AWS Security & IAM',
        prompt:
          'An application running on an AWS EC2 instance needs read-only access to an S3 bucket. What is the recommended security practice?',
        options: [
          'Store root account access keys inside a .env file baked into the AMI.',
          'Attach an IAM Role with a least-privilege S3 read policy directly to the EC2 instance.',
          'Make the S3 bucket public and restrict access by checking the User-Agent header.',
          'Embed temporary STS tokens inside the frontend JavaScript bundle.',
        ],
        correctIndex: 1,
        explanation:
          'IAM instance roles provide short-lived, automatically rotated credentials without storing static keys on disk.',
      },
      {
        id: 'cloud-4',
        topic: 'CI/CD Pipelines',
        prompt:
          'Why should automated linting, unit tests, and vulnerability scans run before the artifact build/push step in a CI/CD pipeline?',
        options: [
          'To fail fast on broken code before spending compute and registry storage on invalid builds.',
          'Because Docker cannot build images unless a unit-test coverage XML file is inside /etc.',
          'To bypass branch protection rules on pull requests.',
          'Because production load balancers execute unit tests on incoming HTTP traffic.',
        ],
        correctIndex: 0,
        explanation:
          'Running fast verification stages first prevents broken commits from producing deployable artifacts.',
      },
      {
        id: 'cloud-5',
        topic: 'Linux & Observability',
        prompt:
          'A containerized service starts returning HTTP 502 behind Nginx after high traffic. Which check best identifies whether the container was OOM-killed?',
        options: [
          'Checking `ping -c 1 localhost` inside the host shell.',
          'Inspecting container exit status / `dmesg` or `kubectl describe pod` for Exit Code 137 (OOMKilled).',
          'Converting the Nginx config from tabs to spaces.',
          'Clearing the browser DNS cache on a client laptop.',
        ],
        correctIndex: 1,
        explanation:
          'Exit Code 137 (SIGKILL) and kernel/pod OOMKilled events directly confirm memory limit exhaustion.',
      },
    ],
  },
  {
    area: 'Data Science & AI/ML',
    keywords: [
      'machine learning', 'deep learning', 'data science', 'data analysis', 'nlp',
      'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
      'power bi', 'tableau', 'llm', 'rag', 'ai', 'ml', 'computer vision',
    ],
    questions: [
      {
        id: 'ds-1',
        topic: 'Preventing Data Leakage',
        prompt:
          'When building a tabular ML pipeline with missing-value imputation and StandardScaler, when should `.fit()` be called?',
        options: [
          'On the combined train + test dataset before splitting so both share identical statistics.',
          'Only on the training split inside cross-validation, then `.transform()` on validation/test splits.',
          'Only on the test set so accuracy on unseen samples is maximized.',
          'After exporting the trained model to ONNX format.',
        ],
        correctIndex: 1,
        explanation:
          'Fitting scalers or imputers on test data leaks distribution information and produces overly optimistic validation scores.',
      },
      {
        id: 'ds-2',
        topic: 'Imbalanced Classification Metrics',
        prompt:
          'You are predicting fraudulent transactions where only 1% of rows are positive. Why is raw Accuracy misleading here?',
        options: [
          'A trivial classifier predicting "not fraud" for every row still achieves 99% accuracy while missing all fraud.',
          'Accuracy cannot be computed for binary classification targets.',
          'Accuracy penalizes false positives twice as heavily as false negatives.',
          'Gradient descent diverges whenever accuracy exceeds 90%.',
        ],
        correctIndex: 0,
        explanation:
          'On skewed classes, Precision-Recall AUC, F1-score, and Recall on the minority class reflect actual detection quality.',
      },
      {
        id: 'ds-3',
        topic: 'Embeddings & Cosine Similarity',
        prompt:
          'Why is cosine similarity widely preferred over raw Euclidean distance when comparing high-dimensional text embeddings?',
        options: [
          'Cosine similarity measures the angle (directional alignment) between vectors rather than sensitivity to vector magnitude.',
          'Cosine similarity guarantees zero false positives without requiring vector normalization.',
          'Euclidean distance only works for 2D coordinates and throws errors above 3 dimensions.',
          'Cosine similarity runs in O(1) time regardless of vocabulary size.',
        ],
        correctIndex: 0,
        explanation:
          'Directional similarity captures semantic overlap even when document lengths or embedding norms differ.',
      },
      {
        id: 'ds-4',
        topic: 'Regularization & Overfitting',
        prompt:
          'A model reaches 99% training accuracy but drops to 64% validation accuracy. Which adjustment directly addresses this?',
        options: [
          'Increase model depth and train for twice as many epochs without early stopping.',
          'Apply stronger regularization (L2/dropout), simplify features, or gather more diverse training data.',
          'Remove the validation split and evaluate strictly on the training set.',
          'Replace ReLU activations with identity linear functions across every layer.',
        ],
        correctIndex: 1,
        explanation:
          'A wide train-validation gap signals overfitting; regularization, pruning capacity, and data augmentation improve generalization.',
      },
      {
        id: 'ds-5',
        topic: 'Pandas & Data Cleaning',
        prompt:
          'In pandas, what is the cleanest way to prevent silent `SettingWithCopyWarning` bugs when filtering a DataFrame and mutating a column?',
        options: [
          'Use chained indexing like `df[df["x"] > 0]["y"] = 1`.',
          'Use explicit `.loc[mask, "y"] = 1` or call `.copy()` before modifying the filtered slice.',
          'Convert the entire DataFrame to Python lists on every row update.',
          'Suppress all Python warnings globally.',
        ],
        correctIndex: 1,
        explanation:
          '`.loc` or an explicit `.copy()` guarantees mutations apply deterministically rather than to an ambiguous view.',
      },
    ],
  },
  {
    area: 'Full-Stack Web Engineering',
    keywords: [
      'react', 'next.js', 'nextjs', 'javascript', 'typescript', 'node.js', 'nodejs',
      'express', 'html', 'css', 'tailwind', 'frontend', 'backend', 'full stack',
      'full-stack', 'rest api', 'graphql', 'mongodb', 'redux',
    ],
    questions: [
      {
        id: 'web-1',
        topic: 'React Rendering & Keys',
        prompt:
          'Why should you avoid using array indices as the `key` prop in a React list where items can be reordered, inserted, or deleted?',
        options: [
          'Array indices cause syntax errors in strict TypeScript.',
          'React reuses component state and DOM nodes by key, causing stale inputs or visual glitches when positions shift.',
          'Using unique IDs disables virtual DOM diffing.',
          'Browser CSS grid layouts ignore elements with string keys.',
        ],
        correctIndex: 1,
        explanation:
          'Stable unique IDs let React track each item identity across insertions, deletions, and sorting.',
      },
      {
        id: 'web-2',
        topic: 'REST API Semantics',
        prompt:
          'Which HTTP status code should an Express/Node endpoint return when a client submits malformed JSON or missing required fields?',
        options: ['200 OK', '302 Found', '400 Bad Request', '503 Service Unavailable'],
        correctIndex: 2,
        explanation:
          '400 Bad Request (or 422 Unprocessable Content) signals a client input validation error rather than a server outage.',
      },
      {
        id: 'web-3',
        topic: 'Async Race Conditions',
        prompt:
          'A search box fires an API fetch on every keystroke. How do you ensure a slower older response never overwrites a newer query result?',
        options: [
          'Store responses inside `localStorage` first.',
          'Pass an `AbortController.signal` to `fetch` and abort the previous in-flight request in effect cleanup.',
          'Use `setTimeout(..., 0)` inside the JSON parser.',
          'Replace HTTPS with HTTP/1.0.',
        ],
        correctIndex: 1,
        explanation:
          'Aborting stale requests (or ignoring out-of-order request IDs) eliminates out-of-order state overwrites.',
      },
      {
        id: 'web-4',
        topic: 'Web Security (XSS)',
        prompt:
          'Why is passing untrusted user input directly into `dangerouslySetInnerHTML` in React dangerous?',
        options: [
          'It triggers Cross-Site Scripting (XSS), allowing injected `<script>` or event handlers to execute in the browser.',
          'It strips all CSS classes from sibling components.',
          'It converts UTF-8 characters into binary.',
          'It blocks CORS preflight requests.',
        ],
        correctIndex: 0,
        explanation:
          'Standard JSX interpolation escapes text safely, whereas raw HTML injection allows arbitrary script execution unless sanitized.',
      },
      {
        id: 'web-5',
        topic: 'Authentication & Secrets',
        prompt:
          'Where should third-party API keys and database connection secrets be stored in a full-stack Vite + Express application?',
        options: [
          'Prefixed with `VITE_` in client source files so React can read them directly.',
          'In server-side environment variables (`process.env`) accessed exclusively by backend `/api/*` routes.',
          'Inside a public `secrets.json` file in `/public`.',
          'Hardcoded in `App.tsx` inside a `useMemo` hook.',
        ],
        correctIndex: 1,
        explanation:
          'Anything bundled by Vite (`VITE_*` or frontend files) is visible to every browser visitor; private secrets must remain server-side.',
      },
    ],
  },
  {
    area: 'Cybersecurity & Networking',
    keywords: [
      'cybersecurity', 'cyber security', 'siem', 'wazuh', 'nmap', 'wireshark',
      'burp suite', 'owasp', 'penetration testing', 'soc', 'firewall', 'networking',
      'tcp/ip', 'vulnerability', 'cryptography', 'infosec', 'splunk',
    ],
    questions: [
      {
        id: 'sec-1',
        topic: 'Injection Defense',
        prompt:
          'What is the most effective primary defense against SQL Injection in a backend web service?',
        options: [
          'Hiding database table names on the frontend.',
          'Using parameterized queries / prepared statements that separate SQL structure from user data.',
          'Base64-encoding user inputs before concatenating them into the SQL string.',
          'Running the web server on a non-standard port.',
        ],
        correctIndex: 1,
        explanation:
          'Parameterized queries treat user inputs strictly as data values, never as executable SQL syntax.',
      },
      {
        id: 'sec-2',
        topic: 'SIEM & Log Triage',
        prompt:
          'A SIEM alert (e.g., Wazuh) shows 400 failed SSH logins from a single external IP followed immediately by one successful login. What should a SOC analyst do first?',
        options: [
          'Ignore the alert because only one login succeeded.',
          'Verify whether the login is legitimate, isolate/block the source if unauthorized, revoke credentials, and inspect post-login process/audit logs.',
          'Delete `/var/log/auth.log` to stop further alerts.',
          'Restart DNS resolution on the client browser.',
        ],
        correctIndex: 1,
        explanation:
          'A brute-force pattern ending in success indicates likely compromise requiring containment, credential rotation, and host investigation.',
      },
      {
        id: 'sec-3',
        topic: 'TLS & Transport Security',
        prompt:
          'During an HTTPS TLS handshake, what does verifying the server certificate chain primarily protect the client against?',
        options: [
          'Hard drive sector corruption on the client device.',
          'Man-in-the-middle impersonation by confirming the server’s public key belongs to the trusted domain.',
          'Excessive RAM usage inside the client browser.',
          'Cross-site request forgery in local cookies.',
        ],
        correctIndex: 1,
        explanation:
          'Certificate validation binds the domain identity to a cryptographic key signed by a trusted Certificate Authority.',
      },
      {
        id: 'sec-4',
        topic: 'Access Control (OWASP)',
        prompt:
          'An API endpoint `/api/invoices/1042` returns another customer’s invoice simply by changing the numeric ID in the URL. Which OWASP Top 10 flaw is this?',
        options: [
          'Broken Access Control (IDOR)',
          'Insufficient Logging & Monitoring',
          'Weak PRNG Seeding',
          'DNS Cache Poisoning',
        ],
        correctIndex: 0,
        explanation:
          'Insecure Direct Object Reference (IDOR) occurs when the server fails to verify ownership/authorization for the requested object ID.',
      },
      {
        id: 'sec-5',
        topic: 'Network Scanning & Ports',
        prompt:
          'In Nmap, why might a host respond differently to a TCP SYN scan (`-sS`) versus a full TCP connect scan (`-sT`)?',
        options: [
          '`-sS` sends only SYN and tears down with RST before completing the 3-way handshake, making it faster and stealthier.',
          '`-sS` scans UDP ports exclusively while `-sT` scans ICMP.',
          '`-sT` decrypts TLS traffic automatically without keys.',
          'Both flags perform identical ICMP echo pings only.',
        ],
        correctIndex: 0,
        explanation:
          'A half-open SYN scan avoids completing the full TCP connection unless elevated raw-socket privileges are unavailable.',
      },
    ],
  },
  {
    area: 'Embedded Systems & IoT',
    keywords: [
      'embedded', 'iot', 'arduino', 'esp32', 'raspberry pi', 'stm32', 'microcontroller',
      'rtos', 'mqtt', 'uart', 'spi', 'i2c', 'verilog', 'vhdl', 'vlsi', 'pcb', 'c++',
    ],
    questions: [
      {
        id: 'iot-1',
        topic: 'Serial Buses (I2C vs SPI)',
        prompt:
          'When connecting multiple sensors to a microcontroller, what is a key electrical difference between I2C and SPI?',
        options: [
          'I2C uses 2 shared open-drain lines (SDA/SCL) with device addresses, whereas SPI uses separate MOSI/MISO/SCLK lines plus a Chip Select per slave.',
          'SPI requires external pull-up resistors to enforce 100 kbps speed.',
          'I2C is always full-duplex while SPI is half-duplex only.',
          'SPI does not use a clock line.',
        ],
        correctIndex: 0,
        explanation:
          'I2C saves pins via 7/10-bit addressing on 2 wires; SPI achieves higher throughput using dedicated data lines and chip-select pins.',
      },
      {
        id: 'iot-2',
        topic: 'Interrupt Service Routines',
        prompt:
          'Why should you avoid calling blocking delays (`delay()`) or heavy `printf`/network calls inside a microcontroller hardware Interrupt Service Routine (ISR)?',
        options: [
          'ISRs block lower-priority interrupts and main loop execution, causing missed events or watchdog resets.',
          'ISRs automatically wipe EEPROM flash memory if they run longer than 1 microsecond.',
          'Compilers prohibit `if` statements inside ISRs.',
          'PWM timers reverse polarity when `delay()` is compiled.',
        ],
        correctIndex: 0,
        explanation:
          'Best practice is to set a `volatile` flag or push to an RTOS queue inside the ISR and handle heavy work in the main task.',
      },
      {
        id: 'iot-3',
        topic: 'IoT Telemetry (MQTT)',
        prompt:
          'In MQTT, what does QoS Level 1 ("At least once") guarantee when publishing sensor readings over an unstable cellular/Wi-Fi link?',
        options: [
          'Messages are sent once and dropped silently if offline.',
          'The broker/receiver acknowledges delivery via PUBACK, retrying until received (possible duplicates if ACK is lost).',
          'End-to-end AES-256 encryption without TLS.',
          'Deterministic sub-millisecond hardware interrupts.',
        ],
        correctIndex: 1,
        explanation:
          'QoS 1 guarantees delivery at least once using PUBACK acknowledgments, trading occasional duplicates for lower overhead than QoS 2.',
      },
      {
        id: 'iot-4',
        topic: 'C / Firmware Memory Safety',
        prompt:
          'Why must a global flag shared between an ISR and the main `while(1)` loop in C be declared with the `volatile` keyword?',
        options: [
          'To allocate the variable inside external SD card storage.',
          'To stop the compiler from caching its value in a CPU register and skipping real memory reads.',
          'To automatically enable mutex locking across multiple CPU cores.',
          'To double the integer bit width from 16 to 32 bits.',
        ],
        correctIndex: 1,
        explanation:
          '`volatile` tells the compiler the value can change asynchronously outside normal control flow, forcing fresh memory loads.',
      },
      {
        id: 'iot-5',
        topic: 'ADC Resolution & Sampling',
        prompt:
          'An ESP32 or STM32 reads an analog temperature sensor with a 12-bit ADC referenced at 3.3V. Approximately how fine is 1 LSB step?',
        options: [
          '~0.8 mV (3.3V / 4096)',
          '~12.9 mV (3.3V / 256)',
          '~330 mV (3.3V / 10)',
          '~3.3 V (3.3V / 1)',
        ],
        correctIndex: 0,
        explanation:
          'A 12-bit ADC divides the reference range into 2^12 = 4096 levels, giving ~0.805 mV per count at 3.3V.',
      },
    ],
  },
  {
    area: 'Core Software & Databases',
    keywords: [
      'python', 'java', 'c++', 'c#', 'sql', 'postgresql', 'mysql', 'sqlite',
      'data structures', 'algorithms', 'oop', 'git', 'backend', 'api', 'software',
    ],
    questions: [
      {
        id: 'core-1',
        topic: 'SQL Indexing & Performance',
        prompt:
          'A table with 5 million rows executes `SELECT * FROM applications WHERE student_email = ?` very slowly. What is the first database optimization to inspect?',
        options: [
          'Adding an index on `student_email` and checking the query plan with `EXPLAIN`.',
          'Converting every column in the table to `TEXT`.',
          'Disabling foreign key constraints across the schema.',
          'Running `SELECT *` twice before rendering.',
        ],
        correctIndex: 0,
        explanation:
          'Without an index, the database performs a full sequential scan; a B-tree index drops lookup cost from O(N) to O(log N).',
      },
      {
        id: 'core-2',
        topic: 'Algorithms & Complexity',
        prompt:
          'You need to check whether any pair of numbers in an unsorted list of 100,000 integers sums to a target `K`. Which approach gives O(N) average time?',
        options: [
          'Nested `for` loops comparing every pair `(i, j)`.',
          'A single pass storing seen values in a hash set and checking if `(K - x)` already exists.',
          'Bubble sort followed by linear scan from index 0.',
          'Recursion branching 3 ways per element without memoization.',
        ],
        correctIndex: 1,
        explanation:
          'Hash-set complement lookup solves Two-Sum in O(N) average time and O(N) space instead of O(N^2).',
      },
      {
        id: 'core-3',
        topic: 'SQL Joins',
        prompt:
          'You want to list all students—including those who have 0 internship applications—along with their application count. Which join should you use?',
        options: [
          'INNER JOIN between `students` and `applications`',
          'LEFT JOIN from `students` to `applications` with `GROUP BY student_id`',
          'CROSS JOIN without any `ON` condition',
          'SELF JOIN on `applications` alone',
        ],
        correctIndex: 1,
        explanation:
          'LEFT JOIN preserves every row from `students` even when no matching rows exist in `applications`.',
      },
      {
        id: 'core-4',
        topic: 'Git Workflows',
        prompt:
          'What is the key difference between `git fetch` and `git pull`?',
        options: [
          '`git fetch` downloads remote commits/refs safely without altering your local working branch, whereas `git pull` fetches and immediately merges/rebases.',
          '`git pull` deletes local untracked files automatically.',
          '`git fetch` pushes local branches to origin.',
          'They are identical aliases.',
        ],
        correctIndex: 0,
        explanation:
          '`git fetch` lets you inspect upstream changes before integrating them into your current branch.',
      },
      {
        id: 'core-5',
        topic: 'API Error Handling & Retries',
        prompt:
          'An external service occasionally returns HTTP 503 (Service Unavailable). Why should client retry logic add exponential backoff rather than looping immediately?',
        options: [
          'Immediate tight-loop retries amplify traffic spikes (thundering herd) and keep the overloaded service down.',
          'HTTP 503 permanently deletes the endpoint if retried under 10 milliseconds.',
          'Backoff automatically converts 503 into 200 OK on the client without hitting the server.',
          'TCP sockets cannot open more than once per minute.',
        ],
        correctIndex: 0,
        explanation:
          'Exponential backoff (with jitter) spreads retries out, giving upstream services room to recover.',
      },
    ],
  },
];

export function detectExpertiseArea(
  resumeText: string,
  matchedSkills: string[] = [],
): { area: ExpertiseArea; detectedSkills: string[] } {
  const combinedCorpus = normalizeCorpus(`${resumeText} ${matchedSkills.join(' ')}`);

  let bestArea: ExpertiseArea = 'Core Software & Databases';
  let bestHits: string[] = [];
  let bestScore = 0;

  for (const domain of DOMAIN_BANK) {
    const hits = domain.keywords.filter((kw) => hasWordBoundaryMatch(combinedCorpus, kw));
    // Give higher specificity domains a slight tie-break over general software
    const specificityBonus = domain.area === 'Core Software & Databases' ? 0 : 0.25;
    const score = hits.length > 0 ? hits.length + specificityBonus : 0;

    if (score > bestScore) {
      bestScore = score;
      bestArea = domain.area;
      bestHits = hits;
    }
  }

  const uniqueSkills = Array.from(
    new Set([
      ...matchedSkills.filter(Boolean),
      ...bestHits.map((k) => k.toUpperCase()),
    ]),
  ).slice(0, 6);

  return {
    area: bestArea,
    detectedSkills: uniqueSkills,
  };
}

export function buildDeterministicExpertiseQuiz(params: {
  resumeText: string;
  matchedSkills?: string[];
}): ExpertiseQuiz {
  const { area, detectedSkills } = detectExpertiseArea(
    params.resumeText,
    params.matchedSkills ?? [],
  );
  const config = DOMAIN_BANK.find((d) => d.area === area) ?? DOMAIN_BANK[DOMAIN_BANK.length - 1];

  return {
    expertiseArea: area,
    detectedSkills,
    source: 'curated',
    questions: config.questions,
  };
}

export function gradeExpertiseQuiz(
  quiz: ExpertiseQuiz,
  selectedAnswers: Record<string, number>,
): ExpertiseTestResult {
  const outcomes: ExpertiseQuestionOutcome[] = quiz.questions.map((q) => {
    const selectedIndex =
      typeof selectedAnswers[q.id] === 'number' ? selectedAnswers[q.id] : -1;
    const isCorrect = selectedIndex === q.correctIndex;
    return {
      id: q.id,
      topic: q.topic,
      prompt: q.prompt,
      selectedIndex,
      correctIndex: q.correctIndex,
      isCorrect,
      selectedOption:
        selectedIndex >= 0 && selectedIndex < q.options.length
          ? q.options[selectedIndex]
          : 'Not answered',
      correctOption: q.options[q.correctIndex] ?? '',
      explanation: q.explanation,
    };
  });

  const correctCount = outcomes.filter((o) => o.isCorrect).length;
  const totalCount = quiz.questions.length;
  const scorePercent = totalCount > 0 ? Math.round((correctCount / totalCount) * 100) : 0;

  const proficiencyBand: ExpertiseTestResult['proficiencyBand'] =
    scorePercent >= 80
      ? 'Strong Mastery'
      : scorePercent >= 60
        ? 'Competent Foundation'
        : 'Needs Focused Review';

  const strongTopics = Array.from(
    new Set(outcomes.filter((o) => o.isCorrect).map((o) => o.topic)),
  );
  const weakTopics = Array.from(
    new Set(outcomes.filter((o) => !o.isCorrect).map((o) => o.topic)),
  );

  return {
    completedAt: new Date().toISOString(),
    expertiseArea: quiz.expertiseArea,
    correctCount,
    totalCount,
    scorePercent,
    proficiencyBand,
    strongTopics,
    weakTopics,
    outcomes,
  };
}
