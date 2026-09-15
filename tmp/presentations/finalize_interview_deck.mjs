import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SKILL_DIR = "/Users/wangyuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const WORKSPACE = "/Users/wangyuhang/Desktop/school/Litho_cpp";
const TEMPLATE = "/Users/wangyuhang/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx";
const CANDIDATE = path.join(WORKSPACE, "tmp/presentations/build/candidate.pptx");
const FINAL_PPTX = path.join(WORKSPACE, "output/presentations/王宇航_技术面试汇报_初稿.pptx");
const RUNTIME_PYTHON = "/Users/wangyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);

const stagingDir = path.join(WORKSPACE, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const requirements = {
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir: WORKSPACE,
  candidatePath: CANDIDATE,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: {
    basis: "design",
    families: ["Helvetica Neue", "PingFang SC"],
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "王宇航_技术面试汇报_初稿.validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, receipt: result.receiptPath ?? null, result }, null, 2));
