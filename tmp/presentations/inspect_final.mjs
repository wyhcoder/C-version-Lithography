import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const p = await PresentationFile.importPptx(await FileBlob.load(
  "/Users/wangyuhang/Desktop/school/Litho_cpp/output/presentations/王宇航_技术面试汇报_初稿.pptx",
));
const result = await p.inspect({
  kind: "slide,notes,image",
  include: "id,slide,title,textPreview,alt",
  maxChars: 20000,
});
console.log(result.ndjson);
