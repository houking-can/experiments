using System;
using System.IO;
using Acrobat;
using System.Reflection;
using System.Windows.Forms;

public class JSObject
{
    private AcroAVDoc g_AVDoc = null;

    public int Convert_pdf_to_html(String fileName, String saveDir)
    {

        FileInfo fileinfo = new FileInfo(fileName);
        if (fileinfo.Exists)
        {
            if (g_AVDoc != null)
            {
                g_AVDoc.Close(0);
            }
            g_AVDoc = new AcroAVDoc();
            g_AVDoc.Open(fileName, "");

        }
        else
        {
            Console.WriteLine("{0} not exist!", fileName);
            return 2;
        }

        if (g_AVDoc.IsValid())
        {
            String savefileName = "";
            if (Directory.Exists(saveDir))
            {
                savefileName = saveDir + '\\' + Path.GetFileNameWithoutExtension(fileName);
            }
            else
            {
                savefileName = Path.GetDirectoryName(saveDir) + Path.GetFileNameWithoutExtension(saveDir);
            }

            CAcroPDDoc pdDoc = (CAcroPDDoc)g_AVDoc.GetPDDoc();
            //Acquire the Acrobat JavaScript Object interface from the PDDoc object
            Object jsObj = pdDoc.GetJSObject();
            Type T = jsObj.GetType();

            object[] saveAsParam1 = { savefileName + ".html", "com.adobe.acrobat.html-3-20" };
            //object[] saveAsParam2 = { savefileName + ".xml", "com.adobe.acrobat.xml-1-00" };
            //object[] saveAsParam3 = { savefileName + ".docx", "com.adobe.acrobat.docx" };
            T.InvokeMember(
                            "saveAs",
                            BindingFlags.InvokeMethod |
                            BindingFlags.Public |
                            BindingFlags.Instance,
                            null, jsObj, saveAsParam1);

            //T.InvokeMember(
            //                "saveAs",
            //                BindingFlags.InvokeMethod |
            //                BindingFlags.Public |
            //                BindingFlags.Instance,
            //                null, jsObj, saveAsParam2);

            //T.InvokeMember(
            //                "saveAs",
            //                BindingFlags.InvokeMethod |
            //                BindingFlags.Public |
            //                BindingFlags.Instance,
            //                null, jsObj, saveAsParam3);

            g_AVDoc.Close(0);
            Console.WriteLine("Convert PDF to XML success.");
            return 0;
        }
        else
        {
            Console.WriteLine("Open {0} failed!", fileName);
            return 3;
        }
        
    }
}


static class Program
{
    static int Main(string[] args)
    {
        String fileName;
        String saveDir = Application.StartupPath;
        if (args.Length == 1)
        {
            fileName = args[0];
        }
        else if(args.Length == 2)
        {
            fileName = args[0];
            saveDir = args[1];
        }
        else
        {
            Console.WriteLine("Usage: PDF2XML.exe inputfile outputdir(optional).");
            return 1;
        }

        int retNumber=0;
        JSObject AdobeJS = new JSObject();
        retNumber = AdobeJS.Convert_pdf_to_html(fileName, saveDir);
        return retNumber;
    }
}
